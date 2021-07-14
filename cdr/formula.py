import re
import math
import ast
import itertools
import numpy as np

from .data import z, c, s, compute_time_mask, expand_impulse_sequence
from .util import names2ix, sn, stderr

interact = re.compile('([^ ]+):([^ ]+)')
spillover = re.compile('(^.+)S([0-9]+)$')
split_irf = re.compile('(.+)\(([^(]+)')
lcg_re = re.compile('[GS](b([0-9]+))?$')
starts_numeric = re.compile('^[0-9]')
non_alphanumeric = re.compile('[^0-9a-zA-Z_]')


def pythonize_string(s):
    """
    Convert string to valid python variable name

    :param s: ``str``; source string
    :return: ``str``; pythonized string
    """

    # Remove whitespace
    s = ''.join(s.split())
    s = non_alphanumeric.sub('_', s)
    return s


def standardize_formula_string(s):
    """
    Standardize a formula string, removing notational variation.
    IRF specifications ``C(...)`` are sorted alphabetically by the IRF call name e.g. ``Gamma()``.
    The order of impulses within an IRF specification is preserved.

    :param s: ``str``; the formula string to be standardized
    :return: ``str``; standardization of **s**
    """

    return str(Formula(s, standardize=False))


def unparse_rangf(t):
    if hasattr(t, 'id'):
        out = t.id
    elif type(t.op).__name__ == 'Mod':
        out = unparse_rangf(t.left) + ':' + unparse_rangf(t.right)
    else:
        raise ValueError('Ill-formed random grouping factor')

    return out


class Formula(object):
    """
    A class for parsing R-style mixed-effects CDR model formula strings and applying them to CDR data matrices.
    
    :param bform_str: ``str``; an R-style mixed-effects CDR model formula string
    """

    IRF_PARAMS = {
        'DiracDelta': [],
        'Exp': ['beta'],
        'ExpRateGT1': ['beta'],
        'Gamma': ['alpha', 'beta'],
        'ShiftedGamma': ['alpha', 'beta', 'delta'],
        'GammaShapeGT1': ['alpha', 'beta'],
        'ShiftedGammaShapeGT1': ['alpha', 'beta', 'delta'],
        'Normal': ['mu', 'sigma2'],
        'SkewNormal': ['mu', 'sigma', 'alpha'],
        'EMG': ['mu', 'sigma', 'beta'],
        'BetaPrime': ['alpha', 'beta'],
        'ShiftedBetaPrime': ['alpha', 'beta', 'delta'],
        'HRFSingleGamma': ['alpha', 'beta'],
        'HRFDoubleGamma1': ['beta'],
        'HRFDoubleGamma2': ['alpha', 'beta'],
        'HRFDoubleGamma3': ['alpha', 'beta', 'c'],
        'HRFDoubleGamma4': ['alpha_main', 'alpha_undershoot', 'beta', 'c'],
        'HRFDoubleGamma5': ['alpha_main', 'alpha_undershoot', 'beta_main', 'beta_undershoot', 'c'],
    }

    CAUSAL_IRFS = {
        'Exp',
        'ExpRateGT1',
        'Gamma',
        'ShiftedGamma',
        'GammaShapeGT1',
        'ShiftedGammaShapeGT1',
        'BetaPrime',
        'ShiftedBetaPrime',
        'HRFSingleGamma',
        'HRFDoubleGamma1',
        'HRFDoubleGamma2',
        'HRFDoubleGamma3',
        'HRFDoubleGamma4',
        'HRFDoubleGamma5',
    }

    LCG_BASES_IX = 2
    LCG_DEFAULT_BASES = 10

    @staticmethod
    def irf_params(family):
        """
        Return list of parameter names for a given IRF family.

        :param family: ``str``; name of IRF family
        :return: ``list`` of ``str``; parameter names
        """

        if family in Formula.IRF_PARAMS:
            out = Formula.IRF_PARAMS[family]
        elif Formula.is_LCG(family):
            bs = Formula.bases(family)
            out = ['x%s' % i for i in range(1, bs + 1)] + ['y%s' % i for i in range(1, bs + 1)] + ['s%s' % i for i in range(1, bs + 1)]
        else:
            out = []

        return out

    @staticmethod
    def is_LCG(family):
        """
        Check whether a kernel is LCG.

        :param family: ``str``; name of IRF family
        :return: ``bool``; whether the kernel is LCG (linear combination of Gaussians)
        """

        return family is not None and lcg_re.match(family) is not None

    @staticmethod
    def bases(family):
        """
        Get the number of bases of a spline kernel.

        :param family: ``str``; name of IRF family
        :return: ``int`` or ``None``; number of bases of spline kernel, or ``None`` if **family** is not a spline.
        """

        if family is None:
            out = None
        else:
            if Formula.is_LCG(family):
                bases = lcg_re.match(family).group(Formula.LCG_BASES_IX)
                if bases is None:
                    bases = Formula.LCG_DEFAULT_BASES
                out = int(bases)
            else:
                out = None
        return out

    @staticmethod
    def prep_formula_string(s):
        return s.strip().replace('.(', '(').replace(':', '%').replace('^', '**')

    @ staticmethod
    def expand_terms(terms):
        new_terms = []
        for t in terms:
            if isinstance(t, list):
                new_terms.append(t)
            else:
                new_terms.append([t])

        return new_terms

    @staticmethod
    def collapse_terms(terms):
        new_terms = []
        for t in terms:
            if isinstance(t, list):
                new_terms += t
            else:
                new_terms.append(t)

        return new_terms

    def __init__(self, bform_str, standardize=True):
        self.build(bform_str, standardize=standardize)
        self.ablated = set()

    def build(self, bform_str, standardize=True):
        """
        Construct internal data from formula string

        :param bform_str: ``str``; source string.
        :return: ``None``
        """

        if standardize:
            bform_str = standardize_formula_string(bform_str)

        self.bform_str = bform_str

        lhs, rhs = bform_str.strip().split('~')

        dv = ast.parse(Formula.prep_formula_string(lhs))
        dv_term = []
        self.process_ast(dv.body[0].value, terms=dv_term, under_irf=True) # Hack: use under_irf=True to prevent function from transforming DV into Dirac Delta IRF
        self.dv_term = [x[0] for x in dv_term]
        self.dv = [x.name() for x in self.dv_term]

        self.has_intercept = {None: True}

        rhs_parsed = ast.parse(Formula.prep_formula_string(rhs))

        self.t = IRFNode()
        if len(rhs_parsed.body):
            self.process_ast(
                rhs_parsed.body[0].value,
                has_intercept=self.has_intercept
            )
        else:
            self.has_intercept[None] = '0' not in [x.strip() for x in rhs.strip().split('+')]

        self.rangf = sorted([x for x in list(self.has_intercept.keys()) if x is not None])

    def process_ast(
            self,
            t,
            terms=None,
            has_intercept=None,
            ops=None,
            rangf=None,
            impulses_by_name=None,
            interactions_by_name=None,
            under_irf=False,
            under_interaction=False
    ):
        """
        Recursively process a node of the Python abstract syntax tree (AST) representation of the formula string and insert data into internal representation of model formula.

        :param t: AST node.
        :param terms: ``list`` or ``None``; CDR terms computed so far, or ``None`` if no CDR terms computed.
        :param has_intercept: ``dict``; map from random grouping factors to boolean values representing whether that grouping factor has a random intercept. ``None`` is used as a key to refer to the population-level intercept.
        :param ops: ``list``; names of ops computed so far, or ``None`` if no ops computed.
        :param rangf: ``str`` or ``None``; name of rangf for random term currently being processed, or ``None`` if currently processing fixed effects portion of model.
        :return: ``None``
        """

        if terms is None:
            terms = []
        if has_intercept is None:
            has_intercept = {}
        if rangf not in has_intercept:
            has_intercept[rangf] = True
        if ops is None:
            ops = []
        if impulses_by_name is None:
            impulses_by_name = {}
        if interactions_by_name is None:
            interactions_by_name = {}
        if type(t).__name__ == 'BinOp':
            if type(t.op).__name__ == 'Add':
                # Delimited terms (impulses and/or IRF calls)

                assert len(ops) == 0, 'Transformation of multiple terms is not supported in CDR formula strings'
                self.process_ast(
                    t.left,
                    terms=terms,
                    has_intercept=has_intercept,
                    ops=ops,
                    rangf=rangf,
                    impulses_by_name=impulses_by_name,
                    interactions_by_name=interactions_by_name,
                    under_irf=under_irf,
                    under_interaction=under_interaction
                )
                self.process_ast(
                    t.right,
                    terms=terms,
                    has_intercept=has_intercept,
                    ops=ops,
                    rangf=rangf,
                    impulses_by_name=impulses_by_name,
                    interactions_by_name=interactions_by_name,
                    under_irf=under_irf,
                    under_interaction=under_interaction
                )

            elif type(t.op).__name__ == 'BitOr':
                # Random effects term
                # LHS: inputs
                # RHS: grouping factor

                assert len(ops) == 0, 'Transformation of random terms is not supported in CDR formula strings'
                assert rangf is None, 'Random terms may not be embedded under other random terms in CDR formula strings'
                self.process_ast(
                    t.left,
                    has_intercept=has_intercept,
                    rangf=unparse_rangf(t.right),
                    impulses_by_name=impulses_by_name,
                    interactions_by_name=interactions_by_name,
                    under_irf=under_irf,
                    under_interaction=under_interaction
                )

            elif type(t.op).__name__ == 'Mod':
                # Interaction term

                subterms = []
                self.process_ast(
                    t.left,
                    terms=subterms,
                    has_intercept=has_intercept,
                    rangf=rangf,
                    impulses_by_name=impulses_by_name,
                    interactions_by_name=interactions_by_name,
                    under_irf=under_irf,
                    under_interaction=True
                )
                self.process_ast(
                    t.right,
                    terms=subterms,
                    has_intercept=has_intercept,
                    rangf=rangf,
                    impulses_by_name=impulses_by_name,
                    interactions_by_name=interactions_by_name,
                    under_irf=under_irf,
                    under_interaction=True
                )

                subterms_list = itertools.product(*subterms)

                new_subterms = []

                for subterms_cur in subterms_list:
                    subterm_irf = [x for x in subterms_cur if (isinstance(x, IRFNode) or isinstance(x, ResponseInteraction))]
                    subterm_nonirf = [x for x in subterms_cur if not (isinstance(x, IRFNode) or isinstance(x, ResponseInteraction))]
                    has_irf_subterm = len(subterm_irf) > 0

                    if has_irf_subterm:
                        assert not under_irf, 'IRF calls cannot be nested in the inputs to another IRF call. To compose IRFs, apply nesting in the impulse response function definition (second argument of IFR call).'

                        if len(subterm_nonirf) > 0:
                            new = ImpulseInteraction(impulses=subterm_nonirf, ops=ops)

                            if new.name() in impulses_by_name:
                                new = impulses_by_name[new.name()]
                            else:
                                impulses_by_name[new.name()] = new

                            if new.name() in interactions_by_name:
                                new = interactions_by_name[new.name()]
                            else:
                                interactions_by_name[new.name()] = new

                            interaction_in = subterm_irf + [new]
                        else:
                            interaction_in = subterm_irf

                        new = ResponseInteraction(
                            interaction_in,
                            rangf=rangf
                        )

                        if new.name() in interactions_by_name:
                            new = interactions_by_name[new.name()]
                        else:
                            interactions_by_name[new.name()] = new

                        if not under_interaction:
                            new.add_rangf(rangf)
                            for irf in [response for response in new.responses() if isinstance(response, IRFNode)]:
                                irf.add_interactions(new)

                        new_subterms.append(new)

                    else:
                        if under_irf or under_interaction:
                            new = ImpulseInteraction(impulses=subterms_cur, ops=ops)

                            if new.name() in impulses_by_name:
                                new = impulses_by_name[new.name()]
                            else:
                                impulses_by_name[new.name()] = new

                            if new.name() in interactions_by_name:
                                new = interactions_by_name[new.name()]
                            else:
                                interactions_by_name[new.name()] = new

                            new_subterms.append(new)
                        else:
                            subterms_name = '%'.join([x.name() for x in subterms_cur])
                            new_str = 'C(%s, DiracDelta())' % subterms_name

                            new_ast = ast.parse(Formula.prep_formula_string(new_str)).body[0].value
                            subterms_cur = []

                            self.process_ast(
                                new_ast,
                                terms=subterms_cur,
                                has_intercept=has_intercept,
                                ops=None,
                                rangf=rangf,
                                impulses_by_name=impulses_by_name,
                                interactions_by_name=interactions_by_name,
                                under_irf=under_irf,
                                under_interaction=under_interaction
                            )
                            for term in subterms_cur:
                                terms.append(term)
                if len(new_subterms) > 0:
                    terms.append(new_subterms)

            elif type(t.op).__name__ == 'Mult':
                # Binary interaction expansion
                # LSH: A
                # RHS: B
                # Output: A + B + A:B

                assert len(ops) == 0, 'Transformation of term expansions is not supported in CDR formula strings'
                subterms_left = []
                self.process_ast(
                    t.left,
                    terms=subterms_left,
                    has_intercept=has_intercept,
                    rangf=rangf,
                    impulses_by_name=impulses_by_name,
                    interactions_by_name=interactions_by_name,
                    under_irf=under_irf,
                    under_interaction=True
                )
                new_subterms_left = []
                for s1 in subterms_left:
                    for s2 in s1:
                        new_subterms_left.append(s2)
                subterms_left = new_subterms_left

                subterms_right = []
                self.process_ast(
                    t.right,
                    terms=subterms_right,
                    has_intercept=has_intercept,
                    rangf=rangf,
                    impulses_by_name=impulses_by_name,
                    interactions_by_name=interactions_by_name,
                    under_irf=under_irf,
                    under_interaction=True
                )
                new_subterms_right = []
                for s1 in subterms_right:
                    for s2 in s1:
                        new_subterms_right.append(s2)
                subterms_right = new_subterms_right

                if under_irf or under_interaction:
                    new_terms = subterms_left + subterms_right
                    for l in subterms_left:
                        for r in subterms_right:
                            new = ImpulseInteraction(impulses=[l,r], ops=ops)

                            if new.name() in impulses_by_name:
                                new = impulses_by_name[new.name()]
                            else:
                                impulses_by_name[new.name()] = new

                            if new.name() in interactions_by_name:
                                new = interactions_by_name[new.name()]
                            else:
                                interactions_by_name[new.name()] = new

                            new_terms.append(new)

                    terms.append(new_terms)

                else:
                    has_irf_subterm = False
                    for x in subterms_left + subterms_right:
                        if type(x).__name__ in ['IRFNode', 'ResponseInteraction']:
                            has_irf_subterm = True
                            break

                    if has_irf_subterm:
                        new_subterms_left = []
                        for s in subterms_left:
                            if isinstance(s, IRFNode):
                                irf = s.irf_to_formula(rangf)
                                new_str = 'C(%s' % s.impulse.name() + ', ' + irf + ')'
                            elif isinstance(s, ResponseInteraction):
                                responses = []
                                for response in s.irf_responses():
                                    irf = response.irf_to_formula(rangf)
                                    responses.append('C(%s' % response.impulse.name() + ', ' + irf + ')')
                                for response in s.non_irf_responses():
                                    responses.append(response.name())
                                new_str = ':'.join(responses)
                            else:
                                new_str = s.name()
                            new_subterms_left.append(new_str)
                        new_subterms_right = []
                        for s in subterms_right:
                            if isinstance(s, IRFNode):
                                irf = s.irf_to_formula(rangf)
                                new_str = 'C(%s' % s.impulse.name() + ', ' + irf + ')'
                            elif isinstance(s, ResponseInteraction):
                                responses = []
                                for response in s.irf_responses():
                                    irf = response.irf_to_formula(rangf)
                                    responses.append('C(%s' % response.impulse.name() + ', ' + irf + ')')
                                for response in s.non_irf_responses():
                                    responses.append(response.name())
                                new_str = ':'.join(responses)
                            else:
                                new_str = s.name()
                            new_subterms_right.append(new_str)

                        subterm_strs = []
                        for l in new_subterms_left:
                            for r in new_subterms_right:
                                subterm_strs.append('%s + %s + %s%%%s' % (l, r, l, r))
                        subterm_str = ' + '.join(subterm_strs)
                    else:
                        new_subterms_left = [x.name() for x in subterms_left]
                        new_subterms_right = [x.name() for x in subterms_right]

                        subterm_strs = []
                        for l in new_subterms_left:
                            for r in new_subterms_right:
                                subterm_strs.append('C(%s + %s + %s%%%s, DiracDelta())' % (l, r, l, r))
                        subterm_str = ' + '.join(subterm_strs)

                    subterm_str = Formula.prep_formula_string(subterm_str)

                    new_ast = ast.parse(subterm_str).body[0].value
                    subterms = []

                    self.process_ast(
                        new_ast,
                        terms=subterms,
                        has_intercept=has_intercept,
                        ops=None,
                        rangf=rangf,
                        impulses_by_name=impulses_by_name,
                        interactions_by_name=interactions_by_name,
                        under_irf=under_irf,
                        under_interaction=under_interaction
                    )
                    terms += subterms

            elif type(t.op).__name__ == 'Pow':
                # N-ary interaction expansion
                # LHS: inputs
                # RHS: n (integer), degree of interaction
                # Output: All interactions of inputs of order up to n

                assert len(ops) == 0, 'Transformation of term expansions is not supported in CDR formula strings'
                subterms = []
                self.process_ast(
                    t.left,
                    terms=subterms,
                    has_intercept=has_intercept,
                    rangf=rangf,
                    impulses_by_name=impulses_by_name,
                    interactions_by_name=interactions_by_name,
                    under_irf=under_irf,
                    under_interaction=True
                )
                new_subterms = []
                for s1 in subterms:
                    for s2 in s1:
                        new_subterms.append(s2)
                subterms = new_subterms

                order = min(int(t.right.n), len(subterms))
                new_terms = []
                for i in range(1, order + 1):
                    collections = itertools.combinations(subterms, i)
                    for tup in collections:
                        if i > 1:
                            new_terms_cur = []
                            for x in tup:
                                if isinstance(x, IRFNode):
                                    irf = x.irf_to_formula(rangf)
                                    name = 'C(%s' % x.impulse.name() + ', ' + irf + ')'
                                    new_terms_cur.append(name)
                                elif isinstance(x, ResponseInteraction):
                                    for response in x.responses():
                                        if isinstance(response, IRFNode):
                                            irf = response.irf_to_formula(rangf)
                                            name = 'C(%s' % response.impulse.name() + ', ' + irf + ')'
                                        else:
                                            name = response.name()
                                        new_terms_cur.append(name)
                                else:
                                    new_terms_cur.append(x.name())
                            new_terms.append('%'.join(new_terms_cur))
                        else:
                            x = tup[0]
                            if isinstance(x, IRFNode):
                                irf = x.irf_to_formula(rangf)
                                name = 'C(%s' % x.impulse.name() + ', ' + irf + ')'
                            elif isinstance(x, ResponseInteraction):
                                for response in x.responses():
                                    if isinstance(response, IRFNode):
                                        irf = response.irf_to_formula(rangf)
                                        name = 'C(%s' % response.impulse.name() + ', ' + irf + ')'
                                    else:
                                        name = response.name()
                            else:
                                name = x.name()
                            new_terms.append(name)

                new_terms_str = ' + '.join(new_terms)
                new_terms_str = Formula.prep_formula_string(new_terms_str)
                subterms = []

                self.process_ast(
                    ast.parse(new_terms_str).body[0].value,
                    terms=subterms,
                    has_intercept=has_intercept,
                    rangf=rangf,
                    impulses_by_name=impulses_by_name,
                    interactions_by_name=interactions_by_name,
                    under_irf=under_irf,
                    under_interaction=under_interaction
                )
                new_subterms = []
                for S in subterms:
                    for s in S:
                        new_subterms.append(s)

                terms.append(new_subterms)

        elif type(t).__name__ == 'Call':
            if t.func.id == 'C':
                # IRF Call
                # Arg 1: Inputs
                # Arg 2: IRF kernel definition

                assert not under_irf, 'IRF calls cannot be nested in the inputs to another IRF call. To compose IRFs, apply nesting in the impulse response function definition (second argument of IFR call).'
                assert len(t.args) == 2, 'C() takes exactly two arguments in CDR formula strings'
                subterms = []
                self.process_ast(
                    t.args[0],
                    terms=subterms,
                    has_intercept=has_intercept,
                    rangf=rangf,
                    impulses_by_name=impulses_by_name,
                    interactions_by_name=interactions_by_name,
                    under_irf=True,
                    under_interaction=under_interaction
                )
                new_subterms = []
                for S in subterms:
                    for s in S:
                        new = self.process_irf(t.args[1], input=s, ops=None, rangf=rangf)
                        new_subterms.append(new)
                terms.append(new_subterms)
            elif t.func.id in Formula.IRF_PARAMS.keys() or lcg_re.match(t.func.id) is not None:
                raise ValueError('IRF calls can only occur as inputs to C() in CDR formula strings')
            else:
                # Unary transform

                assert len(t.args) <= 1, 'Only unary ops on variables supported in CDR formula strings'
                subterms = []
                self.process_ast(
                    t.args[0],
                    terms=subterms,
                    has_intercept=has_intercept,
                    ops=[t.func.id] + ops,
                    rangf=rangf,
                    impulses_by_name=impulses_by_name,
                    interactions_by_name=interactions_by_name,
                    under_irf=under_irf,
                    under_interaction=under_interaction
                )
                terms += subterms

        elif type(t).__name__ in ['Constant', 'Name', 'NameConstant', 'Num']:
            # Basic impulse term

            if type(t).__name__ == 'Name':
                t_id = t.id
            elif type(t).__name__ == 'NameConstant':
                t_id = t.value
            else: # type(t).__name__ == 'Num'
                t_id = str(t.n)
                if t_id in ['0', '1'] and len(ops) == 0:
                    if t_id == '0':
                        has_intercept[rangf] = False
                    return [[t_id]]

            if under_irf or under_interaction:
                new = Impulse(t_id, ops=ops)

                if new.name() in impulses_by_name:
                    new = impulses_by_name[new.name()]
                else:
                    impulses_by_name[new.name()] = new

                terms.append([new])
            else:
                term_name = t_id
                for op in ops:
                    term_name = '%s(%s)' %(op, term_name)

                new_term_str = 'C(%s, DiracDelta())' % term_name

                new_ast = ast.parse(Formula.prep_formula_string(new_term_str)).body[0].value
                subterms = []

                self.process_ast(
                    new_ast,
                    terms=subterms,
                    has_intercept=has_intercept,
                    ops=None,
                    rangf=rangf,
                    impulses_by_name=impulses_by_name,
                    interactions_by_name=interactions_by_name,
                    under_irf=under_irf,
                    under_interaction=under_interaction
                )
                terms += subterms

        else:
            raise ValueError('Operation "%s" is not supported in CDR formula strings' %type(t).__name__)

    def process_irf(
            self,
            t,
            input,
            ops=None,
            rangf=None
    ):
        """
        Process data from AST node representing part of an IRF definition and insert data into internal representation of the model.

        :param t: AST node.
        :param input: ``IRFNode`` object; child IRF of current node
        :param ops: ``list`` of ``str``, or ``None``; ops applied to IRF. If ``None``, no ops applied
        :param rangf: ``str`` or ``None``; name of rangf for random term currently being processed, or ``None`` if currently processing fixed effects portion of model.
        :return: ``IRFNode`` object; the IRF node
        """

        if ops is None:
            ops = []
        assert t.func.id in Formula.IRF_PARAMS.keys() or Formula.is_LCG(t.func.id) is not None, 'Ill-formed model string: process_irf() called on non-IRF node'
        irf_id = None
        coef_id = None
        cont = False
        ranirf = False
        trainable = None
        param_init={}
        if len(t.keywords) > 0:
            for k in t.keywords:
                if k.arg == 'irf_id':
                    if type(k.value).__name__ == 'Str':
                        irf_id = k.value.s
                    elif type(k.value).__name__ == 'Name':
                        irf_id = k.value.id
                    elif type(k.value).__name__ == 'Num':
                        irf_id = str(k.value.n)
                elif k.arg == 'coef_id':
                    if type(k.value).__name__ == 'Str':
                        coef_id = k.value.s
                    elif type(k.value).__name__ == 'Name':
                        coef_id = k.value.id
                    elif type(k.value).__name__ == 'Num':
                        coef_id = str(k.value.n)
                elif k.arg == 'cont':
                    if type(k.value).__name__ == 'Str':
                        if k.value.s in ['True', 'TRUE', 'true', 'T']:
                            cont = True
                    elif type(k.value).__name__ == 'Name':
                        if k.value.id in ['True', 'TRUE', 'true', 'T']:
                            cont = True
                    elif type(k.value).__name__ == 'NameConstant':
                        cont = k.value.value
                    elif type(k.value).__name__ == 'Num':
                        cont = k.value.n > 0
                elif k.arg == 'ran':
                    if type(k.value).__name__ == 'Str':
                        if k.value.s in ['True', 'TRUE', 'true', 'T']:
                            ranirf = True
                    elif type(k.value).__name__ == 'Name':
                        if k.value.id in ['True', 'TRUE', 'true', 'T']:
                            ranirf = True
                    elif type(k.value).__name__ == 'NameConstant':
                        ranirf = k.value.value
                    elif type(k.value).__name__ == 'Num':
                        ranirf = k.value.n > 0
                elif k.arg == 'trainable':
                    assert type(k.value).__name__ == 'List', 'Non-list argument provided to keyword arg "trainable"'
                    trainable = []
                    for x in k.value.elts:
                        if type(x).__name__ == 'Name':
                            trainable.append(x.id)
                        elif type(x).__name__ == 'Str':
                            trainable.append(x.s)
                else:
                    if type(k.value).__name__ == 'Num':
                        param_init[k.arg] = k.value.n
                    elif type(k.value).__name__ == 'UnaryOp':
                        assert type(k.value.op).__name__ == 'USub', 'Invalid operator provided to to IRF parameter "%s"' %k.arg
                        assert type(k.value.operand).__name__ == 'Num', 'Non-numeric initialization provided to IRF parameter "%s"' %k.arg
                        param_init[k.arg] = -k.value.operand.n
                    else:
                        raise ValueError('Non-numeric initialization provided to IRF parameter "%s"' %k.arg)

        if isinstance(input, IRFNode):
            new = IRFNode(
                family=t.func.id,
                irfID=irf_id,
                ops=ops,
                fixed=rangf is None,
                rangf=rangf if ranirf else None,
                param_init=param_init,
                trainable=trainable
            )

            new.add_child(input)

            if len(t.args) > 0:
                assert len(t.args) == 1, 'Ill-formed model string: IRF can take at most 1 positional argument'
                p = self.process_irf(
                    t.args[0],
                    input = new,
                    rangf=rangf
                )
            else:
                p = self.t

            p.add_child(new)

        else:
            new = IRFNode(
                family='Terminal',
                impulse=input,
                coefID=coef_id,
                cont=cont,
                fixed=rangf is None,
                rangf=rangf,
                param_init=param_init,
                trainable=trainable
            )

            p = self.process_irf(
                t,
                input=new,
                rangf=rangf
            )

        for c in p.children:
            if c.local_name() == new.local_name():
                new = c
                break

        return new

    def responses(self):
        """
        Get list of modeled response variables.

        :return: ``list`` of ``Impulse``; modeled response variables.
        """

        return self.dv_term

    def response_names(self):
        """
        Get list of names modeled response variables.

        :return: ``list`` of ``str``; names modeled response variables.
        """

        return [x.name() for x in self.dv_term]

    def apply_op(self, op, arr):
        """
        Apply op **op** to array **arr**.

        :param op: ``str``; name of op.
        :param arr: ``numpy`` or ``pandas`` array; source data.
        :return: ``numpy`` array; transformed data.
        """

        arr.fillna(0, inplace=True)
        arr[arr == np.inf] = 0
        if op in ['c', 'c.']:
            out = c(arr)
        elif op in ['z', 'z.']:
            out = z(arr)
        elif op in ['s', 's.']:
            out = s(arr)
        elif op == 'log':
            out = np.log(arr)
        elif op == 'log1p':
            out = np.log(arr + 1)
        elif op == 'exp':
            out = np.exp(arr)
        else:
            raise ValueError('Unrecognized op: "%s".' % op)
        return out

    def apply_ops(self, impulse, X):
        """
        Apply all ops defined for an impulse

        :param impulse: ``Impulse`` object; the impulse.
        :param X: list of ``pandas`` tables; table containing the impulse data.
        :return: ``pandas`` table; table augmented with transformed impulse.
        """

        if not isinstance(X, list):
            X = [X]
            delistify = True
        else:
            delistify = False

        for i in range(len(X)):
            _X = X[i]
            ops = impulse.ops

            expanded_impulses = None
            if impulse.id not in _X:
                if type(impulse).__name__ == 'ImpulseInteraction':
                    _X, expanded_impulses, expanded_atomic_impulses = impulse.expand_categorical(_X)
                    for x in expanded_atomic_impulses:
                        for a in x:
                            _X = self.apply_ops(a, _X)
                    for x in expanded_impulses:
                        if x.name() not in _X:
                            _X[x.id] = _X[[y.name() for y in x.atomic_impulses]].product(axis=1)
            else:
                if type(impulse).__name__ == 'ImpulseInteraction':
                    _X, expanded_impulses, _ = impulse.expand_categorical(_X)
                else:
                    _X, expanded_impulses = impulse.expand_categorical(_X)

            if expanded_impulses is not None:
                for x in expanded_impulses:
                    if x.name() not in _X:
                        new_col = _X[x.id]
                        for j in range(len(ops)):
                            op = ops[j]
                            new_col = self.apply_op(op, new_col)
                        _X[x.name()] = new_col

            X[i] = _X

        if delistify:
            X = X[0]

        return X

    def compute_2d_predictor(
            self,
            predictor_name,
            X,
            first_obs,
            last_obs,
            history_length=128,
            future_length=None,
            minibatch_size=50000
    ):
        """
        Compute 2D predictor (predictor whose value depends on properties of the most recent impulse).

        :param predictor_name: ``str``; name of predictor
        :param X: ``pandas`` table; input data
        :param first_obs: ``pandas`` ``Series`` or 1D ``numpy`` array; row indices in ``X`` of the start of the series associated with each regression target.
        :param last_obs: ``pandas`` ``Series`` or 1D ``numpy`` array; row indices in ``X`` of the most recent observation in the series associated with each regression target.
        :param minibatch_size: ``int``; minibatch size for computing predictor, can help with memory footprint
        :return: 2-tuple; new predictor name, ``numpy`` array of predictor values
        """

        supported = [
            'cosdist2D',
            'eucldist2D'
        ]

        assert predictor_name in supported, '2D predictor "%s" not currently supported' %predictor_name

        window_length = history_length + future_length

        if predictor_name in ['cosdist2D', 'eucldist2D']:
            is_embedding_dimension = re.compile('d([0-9]+)')

            embedding_colnames = [c for c in X.columns if is_embedding_dimension.match(c)]
            embedding_colnames = sorted(embedding_colnames, key=lambda x: float(x[1:]))

            assert len(embedding_colnames) > 0, 'Model formula contains vector distance predictors but no embedding columns found in the input data'

            new_2d_predictor = []
            if predictor_name == 'cosdist2D':
                new_2d_predictor_name = 'cosdist2D'
                stderr('Computing pointwise cosine distances...\n')
            elif predictor_name == 'eucldist2D':
                stderr('Computing pointwise Euclidean distances...\n')
                new_2d_predictor_name = 'eucldist2D'

            for i in range(0, len(first_obs), minibatch_size):
                stderr('\rProcessing batch %d/%d' %(i/minibatch_size + 1, math.ceil(float(len(first_obs))/minibatch_size)))
                X_embeddings, _, _ = expand_impulse_sequence(
                    np.array(X[embedding_colnames]),
                    np.array(X['time']),
                    np.array(first_obs)[i:i+minibatch_size],
                    np.array(last_obs)[i:i+minibatch_size],
                    window_length,
                    fill=np.nan
                )
                X_bases = X_embeddings[:, -1:, :]

                if predictor_name == 'cosdist2D':
                    numerator = (X_bases * X_embeddings).sum(axis=2)
                    denominator = np.sqrt((X_bases ** 2).sum(axis=2)) * np.sqrt((X_embeddings ** 2).sum(axis=2))
                    cosine_distances = numerator / (denominator)

                    cosine_distances = np.where(np.isfinite(cosine_distances), cosine_distances,
                                                np.zeros(X_embeddings.shape[:2]))

                    new_2d_predictor.append(np.expand_dims(cosine_distances, -1))

                elif predictor_name == 'eucldist2D':
                    diffs = X_bases - X_embeddings

                    euclidean_distances = np.sqrt((diffs ** 2).sum(axis=2))
                    euclidean_distances = np.where(np.isfinite(euclidean_distances), euclidean_distances,
                                                   np.zeros(X_embeddings.shape[:2]))

                    new_2d_predictor.append(np.expand_dims(euclidean_distances, -1))

            stderr('\n')
            new_2d_predictor = np.concatenate(new_2d_predictor, axis=0)

        return new_2d_predictor_name, new_2d_predictor

    def apply_op_2d(self, op, arr, time_mask):
        """
        Apply op to 2D predictor (predictor whose value depends on properties of the response).

        :param op: ``str``; name of op.
        :param arr: ``numpy`` or array; source data.
        :param time_mask: ``numpy`` array; mask for padding cells
        :return: ``numpy`` array; transformed data
        """

        with np.errstate(invalid='ignore'):
            n = time_mask.sum()
            time_mask = time_mask[..., None]
            if op in ['c', 'c.']:
                mean = arr.sum() / n
                out = (arr - mean) * time_mask
            elif op in ['z', 'z.']:
                mean = arr.sum() / n
                sd = np.sqrt((((arr - mean) ** 2) * time_mask).sum() / n)
                out = ((arr - mean) / sd) * time_mask
            elif op in ['s', 's.']:
                mean = arr.sum() / n
                sd = np.sqrt((((arr - mean) ** 2) * time_mask).sum() / n)
                out = (arr / sd) * time_mask
            elif op == 'log':
                out = np.where(time_mask, np.log(arr), np.zeros_like(arr))
            elif op == 'log1p':
                out = np.where(time_mask, np.log(arr + 1), np.zeros_like(arr))
            elif op == 'exp':
                out = np.where(time_mask, np.exp(arr), np.zeros_like(arr))
            else:
                raise ValueError('Unrecognized op: "%s".' % op)
            return out

    def apply_ops_2d(self, impulse, X_2d_predictor_names, X_2d_predictors, time_mask):
        """
        Apply all ops defined for a 2D predictor (predictor whose value depends on properties of the response).

        :param impulse: ``Impulse`` object; the impulse.
        :param X_2d_predictor_names: ``list`` of ``str``; names of 2D predictors.
        :param X_2d_predictors: ``numpy`` array; source data.
        :param time_mask: ``numpy`` array; mask for padding cells
        :return: 2-tuple; ``list`` of new predictor name, ``numpy`` array of predictor values
        """

        assert time_mask is not None, 'Trying to compute 2D predictor but no time mask provided'
        ops = impulse.ops
        if impulse.name() not in X_2d_predictor_names:
            if impulse.id not in X_2d_predictor_names:
                raise ValueError('Unrecognized term "%s" in model formula' %impulse.id)
            else:
                i = names2ix(impulse.id, X_2d_predictor_names)[0]
                new_2d_predictor = X_2d_predictors[:, :, i:i + 1]
            for i in range(len(ops)):
                op = ops[i]
                new_2d_predictor = self.apply_op_2d(op, new_2d_predictor, time_mask)
            X_2d_predictor_names.append(impulse.name())
            X_2d_predictors = np.concatenate([X_2d_predictors, new_2d_predictor], axis=2)
        return X_2d_predictor_names, X_2d_predictors

    def apply_formula(
            self,
            X,
            Y,
            X_in_Y_names=None,
            all_interactions=False,
            series_ids=None
    ):
        """
        Extract all data and compute all transforms required by the model formula.

        :param X: list of ``pandas`` tables; impulse data.
        :param Y: list of ``pandas`` tables; response data.
        :param X_in_Y_names: ``list`` or ``None``; List of column names for response-aligned predictors (predictors measured for every response rather than for every input) if applicable, ``None`` otherwise.
        :param all_interactions: ``bool``; add powerset of all conformable interactions.
        :param series_ids: ``list`` of ``str`` or ``None``; list of ids to use as grouping factors for lagged effects. If ``None``, lagging will not be attempted.
        :return: triple; transformed **X**, transformed **y**, response-aligned predictor names
        """

        if not isinstance(X, list):
            X = [X]

        for dv in self.dv_term:
            found = False
            for i, _Y in enumerate(Y):
                if dv.id in _Y:
                    found = True
                    if dv.name() not in _Y:
                        _Y = self.apply_ops(dv, _Y)
                    Y[i] = _Y
                    break
            assert found, 'Response variable %s not found in input data.' % dv.name()
        impulses = self.t.impulses(include_interactions=True)

        if all_interactions:
            impulse_names = self.t.impulse_names(include_interactions=False)
            interaction_names = set(self.t.impulse_names(include_interactions=True)) - set(impulse_names)
            atomic_impulses = self.t.impulses(include_interactions=False)
            extra_interactions = set(
                itertools.chain.from_iterable(
                    itertools.combinations(atomic_impulses, n) for n in range(2, len(atomic_impulses) + 1)
                )
            )
            for x in extra_interactions:
                name = ':'.join(impulse.name() for impulse in x)
                if name not in interaction_names:
                    impulses.append(ImpulseInteraction(x))

        impulses = sorted(list(set(impulses)), key=lambda x: x.name())

        X_columns = set()
        for _X in X:
            for c in _X.columns:
                X_columns.add(c)

        for impulse in impulses:
            if type(impulse).__name__ == 'ImpulseInteraction':
                to_process = impulse.impulses()
            else:
                to_process = [impulse]

            for x in to_process:
                if x.id in X_columns:
                    for i in range(len(X)):
                        _X = X[i]
                        if x.id in _X:
                            _X = self.apply_ops(x, _X)
                            X[i] = _X
                            break
                else: # Not in X, so either it's spilled over (legacy from Cognition expts) or it's in Y (response aligned)
                    sp = spillover.match(x.id)
                    if sp and sp.group(1) in X_columns and series_ids is not None:
                        x_id = sp.group(1)
                        n = int(sp.group(2))
                        for i in range(len(X)):
                            _X = X[i]
                            if x_id in _X:
                                _X[x_id] = _X.groupby(series_ids)[x_id].shift_activations(n, fill_value=0.)
                                _X = self.apply_ops(x, _X)
                                X[i] = _X
                                break
                    else: # Response aligned
                        for i, _Y in enumerate(Y):
                            assert x.id in _Y, 'Impulse %s not found in data. Either it is missing from all of the predictor files X, or (if response aligned) it is missing from at least one of the response files Y.' % x.name()
                            Y[i] = self.apply_ops(x, _Y)
                        if X_in_Y_names is None:
                            X_in_Y_names = []
                        X_in_Y_names.append(x.name())

            if type(impulse).__name__ == 'ImpulseInteraction':
                response_aligned = False
                for x in impulse.impulses():
                    if x.id not in X_columns:
                        response_aligned = True
                        break
                if response_aligned:
                    for i, _Y in enumerate(Y):
                        Y[i] = self.apply_ops(impulse, _Y)
                    if X_in_Y_names is None:
                        X_in_Y_names = []
                    X_in_Y_names.append(impulse.name())
                else:
                    found = False
                    for i in range(len(X)):
                        _X = X[i]
                        in_X = True
                        for atom in impulse.impulses():
                            if atom.id not in _X:
                                in_X = False
                        if in_X:
                            _X = self.apply_ops(impulse, _X)
                            X[i] = _X
                            found = True
                            break
                    if not found:
                        raise ValueError('No single predictor file contains all features in ImpulseInteraction, and interaction across files is not possible because of asynchrony. Consider interacting the responses, rather than the impulses.')

        for i in range(len(X)):
            _X = X[i]
            for col in [x for x in _X.columns if spillover.match(x)]:
                _X[col] = _X[col].fillna(0)
            X[i] = _X

        for _Y in Y:
            for gf in self.rangf:
                gf_s = gf.split(':')
                if len(gf_s) > 1 and gf not in _Y:
                    _Y[gf] = _Y[gf_s].agg(lambda x: '_'.join([str(_x) for _x in x]), axis=1)

        return X, Y, X_in_Y_names

    def ablate_impulses(self, impulse_ids):
        """
        Remove impulses in **impulse_ids** from fixed effects (retaining in any random effects).

        :param impulse_ids: ``list`` of ``str``; impulse ID's
        :return: ``None``
        """

        if not isinstance(impulse_ids, list):
            impulse_ids = [impulse_ids]
        self.t.ablate_impulses(impulse_ids)
        self.ablated |= set(impulse_ids)

    def unablate_impulses(self, impulse_ids):
        """
        Insert impulses in **impulse_ids** into fixed effects (leaving random effects structure unchanged).

        :param impulse_ids: ``list`` of ``str``; impulse ID's
        :return: ``None``
        """

        if not isinstance(impulse_ids, list):
            impulse_ids = [impulse_ids]
        self.t.unablate_impulses(impulse_ids)
        self.ablated -= set(impulse_ids)

    def remove_impulses(self, impulse_ids):
        """
        Remove impulses in **impulse_ids** from the model (both fixed and random effects).

        :param impulse_ids: ``list`` of ``str``; impulse ID's
        :return:  ``None``
        """

        if not isinstance(impulse_ids, list):
            impulse_ids = [impulse_ids]
        self.t.remove_impulses(impulse_ids)

    def insert_impulses(self, impulses, irf_str, rangf=None):
        """
        Insert impulses in **impulse_ids** into fixed effects and all random terms.

        :param impulse_ids: ``list`` of ``str``; impulse ID's
        :return: ``None``
        """

        if not isinstance(impulses, list):
            impulses = [impulses]

        if rangf is None:
            rangf = []
        elif not isinstance(rangf, list):
            rangf = [rangf]

        rangf = sorted(rangf)

        bform = str(self)
        bform += ' + C(' + ' + '.join(impulses) + ', ' + irf_str + ')'
        for gf in rangf:
            bform += ' + (C(' + ' + '.join(impulses) + ', ' + irf_str + ') | ' + gf + ')'

        self.build(bform)

    def to_lmer_formula_string(self, z=False, correlated=True):
        """
        Generate an ``lme4``-style LMER model string representing the structure of the current CDR model.
        Useful for 2-step analysis in which data are transformed using CDR, then fitted using LME.

        :param z: ``bool``; z-transform convolved predictors.
        :param correlated: ``bool``; whether to use correlated random intercepts and slopes.
        :return: ``str``; the LMER formula string.
        """

        assert len(self.dv_term) == 1, 'Models with multivariate responses cannot be rendered in an LMER formula.'

        fixed = []
        random = {}

        if correlated:
            ranef_op = ' | '
        else:
            ranef_op = ' || '

        for terminal in self.t.terminals():
            if terminal.p.irfID is None:
                name = sn('-'.join(terminal.name().split('-')[:-1]))
            else:
                name = sn(terminal.name())
            if z:
                name = 'z.(' + name + ')'

            if terminal.fixed:
                fixed.append(name)
            for gf in terminal.rangf:
                if gf not in random:
                    random[gf] = []
                random[gf].append(name)

        for x in self.t.interactions():
            subterm_strings = []
            for y in x.atomic_responses:
                if isinstance(y, IRFNode) and y.p.irfID is None:
                    new_term_string = sn('-'.join(y.name().split('-')[:-1]))
                else:
                    new_term_string = sn(y.name())
                subterm_strings.append(new_term_string)
            subterm_strings = ':'.join(subterm_strings)
            if z:
                subterm_strings = 'z.(' + subterm_strings + ')'
            if None in x.rangf:
                fixed.append(subterm_strings)
            for gf in sorted(list(random.keys())):
                if gf in x.rangf:
                    random[gf].append(subterm_strings)

        out = str(self.dv_term[0]) + ' ~ '

        if not self.has_intercept[None]:
            out += '0 + '

        out += ' + '.join([x for x in fixed])

        for gf in sorted(list(random.keys())):
            out += ' + (' + ('1 + ' if self.has_intercept[gf] else '0 + ') + ' + '.join([x for x in random[gf]]) + ranef_op + gf + ')'

        for gf in sorted(list(self.has_intercept.keys()), key=lambda x: (x is None, x)):
            if gf is not None and gf not in random and self.has_intercept[gf]:
                out += ' + (1 | ' + gf + ')'

        return out

    def _rhs_str(self, t=None):
        if t == None:
            t = self.t
        out = ''

        term_strings = []

        if not self.has_intercept[None]:
            term_strings.append('0')

        terms = t.formula_terms()

        if None in terms:
            fixed = terms.pop(None)
            new_terms = {}
            for term in fixed:
                if term['irf'] in new_terms:
                    new_terms[term['irf']]['impulses'] += term['impulses']
                else:
                    new_terms[term['irf']] = term
            new_terms = [new_terms[x] for x in sorted(list(new_terms.keys()))]
            term_strings.append(' + '.join(['C(%s, %s)' %(' + '.join([x.name() for x in y['impulses']]), y['irf']) for y in new_terms]))

            for x in t.interactions():
                if None in x.rangf:
                    subterm_strings = []
                    for y in x.atomic_responses:
                        if isinstance(y, IRFNode):
                            irf = y.irf_to_formula(None)
                            new_term_string = 'C(%s' % y.impulse.name() + ', ' + irf + ')'
                        else:
                            new_term_string = y.name()
                        subterm_strings.append(new_term_string)
                    term_strings.append(':'.join(subterm_strings))

        for rangf in sorted(list(terms.keys())):
            ran = terms[rangf]
            new_terms = {}
            for term in ran:
                if term['irf'] in new_terms:
                    new_terms[term['irf']]['impulses'] += term['impulses']
                else:
                    new_terms[term['irf']] = term
            new_terms = [new_terms[x] for x in sorted(list(new_terms.keys()))]
            new_terms_str = '('
            if not self.has_intercept[rangf]:
                new_terms_str += '0 + '

            interactions_str = ''
            for x in t.interactions():
                if rangf in x.rangf:
                    subterm_strings = []
                    for y in x.atomic_responses:
                        if isinstance(y, IRFNode):
                            irf = y.irf_to_formula(rangf)
                            new_term_string = 'C(%s' % y.impulse.name() + ', ' + irf + ')'
                        else:
                            new_term_string = y.name()
                        subterm_strings.append(new_term_string)
                    interactions_str += ' + ' + ':'.join(subterm_strings)

            new_terms_str += ' + '.join(['C(%s, %s)' % (' + '.join([x.name() for x in y['impulses']]), y['irf']) for y in new_terms]) + interactions_str + ' | %s)' %rangf
            term_strings.append(new_terms_str)

        ran_intercepts = []
        for key in sorted(list(self.has_intercept.keys()), key=lambda x: (x is None, x)):
            if key is not None and not key in terms and self.has_intercept[key]:
                ran_intercepts.append(key)
        if ran_intercepts:
            term_strings.append(' + '.join(['(1 | %s)' % key for key in ran_intercepts]))

        out += ' + '.join(term_strings)

        return out

    def to_string(self, t=None):
        """
        Stringify the formula, using **t** as the RHS.

        :param t: ``IRFNode`` or ``None``; IRF node to use as RHS. If ``None``, uses root IRF associated with ``Formula`` instance.
        :return: ``str``; stringified formula.
        """
        if t == None:
            t = self.t

        out = ' + '.join([str(x) for x in self.dv_term]) + ' ~ ' + self._rhs_str(t=t)

        return out

    def pc_transform(self, n_pc, pointers=None):
        """
        Get transformed formula with impulses replaced by principal components.

        :param n_pc: ``int``; number of principal components in transform.
        :param pointers: ``dict``; map from source nodes to transformed nodes.
        :return: ``list`` of ``IRFNode``; tree forest representing current state of the transform.
        """

        new_t = self.t.pc_transform(n_pc, pointers=pointers)[0]
        new_formstring = self.to_string(t=new_t)
        new_form = Formula(new_formstring)
        return new_form

    def categorical_transform(self, X):
        """
        Get transformed formula with categorical predictors in **X** expanded.

        :param X: list of ``pandas`` tables; input data.
        :return: ``Formula``; transformed ``Formula`` object
        """

        new_t = self.t.categorical_transform(X)[0]
        new_formstring = self.to_string(t=new_t)
        new_form = Formula(new_formstring)
        return new_form

    def __str__(self):
        return self.to_string()



class Impulse(object):
    """
    Data structure representing an impulse in a CDR model.

    :param name: ``str``; name of impulse
    :param ops: ``list`` of ``str``, or ``None``; ops to apply to impulse. If ``None``, no ops.
    """

    def __init__(self, name, ops=None):
        if ops is None:
            ops = []
        self.ops = ops[:]
        self.name_str = name
        for op in self.ops:
            self.name_str = op + '(' + self.name_str + ')'
        self.id = name

    def __str__(self):
        return self.name_str

    def name(self):
        """
        Get name of term.

        :return: ``str``; name.
        """

        return self.name_str

    def categorical(self, X):
        """
        Checks whether impulse is categorical in a dataset

        :param X: list ``pandas`` tables; data to to check.
        :return: ``bool``; ``True`` if impulse is categorical in **X**, ``False`` otherwise.
        """

        if not isinstance(X, list):
            X = [X]

        for _X in X:
            if self.id in _X:
                dtype = _X[self.id].dtype
                if dtype.name == 'category' or not np.issubdtype(dtype, np.number):
                    return True
        
        return False

    def expand_categorical(self, X):
        """
        Expand any categorical predictors in **X** into 1-hot columns.

        :param X: list of ``pandas`` tables; input data
        :return: 2-tuple of ``pandas`` table, ``list`` of ``Impulse``; expanded data, list of expanded ``Impulse`` objects
        """

        if not isinstance(X, list):
            X = [X]
            delistify = True
        else:
            delistify = False

        impulses = [self]

        for i in range(len(X)):
            _X = X[i]
            if self.id in _X and self.categorical(X):
                vals = sorted(_X[self.id].unique())[1:]
                impulses = [Impulse('_'.join([self.id, pythonize_string(str(val))]), ops=self.ops) for val in vals]
                expanded_value_names = [str(val) for val in vals]
                for j in range(len(impulses)):
                    x = impulses[j]
                    val = expanded_value_names[j]
                    if x.id not in _:
                        _X[x.id] = (_X[self.id] == val).astype('float')
                X[i] = _X
                break

        if delistify:
            X = X[0]

        return X, impulses

class ImpulseInteraction(object):
    """
    Data structure representing an interaction of impulse-aligned variables (impulses) in a CDR model.

    :param impulses: ``list`` of ``Impulse``; impulses to interact.
    :param ops: ``list`` of ``str``, or ``None``; ops to apply to interaction. If ``None``, no ops.
    """

    def __init__(self, impulses, ops=None):
        if ops is None:
            ops = []
        self.ops = ops[:]
        self.atomic_impulses = []
        names = set()
        for x in impulses:
            if isinstance(x, ImpulseInteraction):
                for impulse in x.impulses():
                    names.add(impulse.name())
                    self.atomic_impulses.append(impulse)
            else:
                names.add(x.name())
                self.atomic_impulses.append(x)
        self.name_str = ':'.join([x.name() for x in sorted(impulses, key=lambda x: x.name())])
        for op in self.ops:
            self.name_str = op + '(' + self.name_str + ')'
        self.id = ':'.join([x.id for x in sorted(self.atomic_impulses, key=lambda x: x.id)])

    def __str__(self):
        return self.name_str

    def name(self):
        """
        Get name of interation impulse.

        :return: ``str``; name.
        """

        return self.name_str

    def impulses(self):
        """
        Get list of impulses dominated by interaction.

        :return: ``list`` of ``Impulse``; impulses dominated by interaction.
        """

        return self.atomic_impulses

    def expand_categorical(self, X):
        """
        Expand any categorical predictors in **X** into 1-hot columns.

        :param X: list of ``pandas`` tables; input data.
        :return: 3-tuple of ``pandas`` table, ``list`` of ``ImpulseInteraction``, ``list`` of ``list`` of ``Impulse``; expanded data, list of expanded ``ImpulseInteraction`` objects, list of lists of expanded ``Impulse`` objects, one list for each interaction.
        """

        if not isinstance(X, list):
            X = [X]
            delistify = True
        else:
            delistify = False

        expanded_atomic_impulses = []
        for x in self.impulses():
            X, expanded_atomic_impulses_cur = x.expand_categorical(X)
            expanded_atomic_impulses.append(expanded_atomic_impulses_cur)
        expanded_interaction_impulses = [ImpulseInteraction(x, ops=self.ops) for x in itertools.product(*expanded_atomic_impulses)]

        if delistify:
            X = X[0]

        return X, expanded_interaction_impulses, expanded_atomic_impulses

class ResponseInteraction(object):
    """
    Data structure representing an interaction of response-aligned variables (containing at least one IRF-convolved impulse) in a CDR model.

    :param responses: ``list`` of terminal ``IRFNode``, ``Impulse``, and/or ``ImpulseInteraction`` objects; responses to interact.
    :param rangf: ``str`` or list of ``str``; random grouping factors for which to build random effects for this interaction.
    """

    def __init__(self, responses, rangf=None):
        self.atomic_responses = []
        for x in responses:
            assert (type(x).__name__ == 'IRFNode' and x.terminal()) or type(x).__name__ in ['Impulse', 'ImpulseInteraction', 'ResponseInteraction'], 'All inputs to ResponseInteraction must be either terminal IRFNode, Impulse, ImpulseInteraction, or ResponseInteraction objects. Got %s.' % type(x).__name__
            if isinstance(x, ResponseInteraction):
                for y in x.responses():
                    self.atomic_responses.append(y)
            else:
                self.atomic_responses.append(x)
        self.name_str = '|'.join([x.name() for x in sorted(responses, key=lambda x: x.name())])
        if not isinstance(rangf, list):
            rangf = [rangf]
        self.rangf = rangf

    def __str__(self):
        return self.name_str

    def name(self):
        """
        Get name of interation impulse.

        :return: ``str``; name.
        """

        return self.name_str

    def responses(self):
        """
        Get list of variables dominated by interaction.

        :return: ``list`` of ``IRFNode``, ``Impulse``, and/or ``ImpulseInteraction`` objects; impulses dominated by interaction.
        """

        return self.atomic_responses[:]

    def irf_responses(self):
        """
        Get list of IRFs dominated by interaction.

        :return: ``list`` of ``IRFNode`` objects; terminal IRFs dominated by interaction.
        """

        return [x for x in self.atomic_responses if type(x).__name__ == 'IRFNode']

    def non_irf_responses(self):
        """
        Get list of non-IRF response-aligned variables dominated by interaction.

        :return: ``list`` of ``Impulse`` and/or ``ImpulseInteraction`` objects; non-IRF variables dominated by interaction.
        """

        return [x for x in self.atomic_responses if type(x).__name__ != 'IRFNode']

    def contains_member(self, x):
        """
        Check if object is a member of the set of responses belonging to this interaction

        :param x: ``IRFNode``, ``Impulse``, and/or ``ImpulseInteraction`` object; object to check.
        :return: ``bool``; whether x is a member of the set of responses
        """

        out = False
        if type(x).__name__ == 'IRFNode' and x.terminal() or type(x).__name__ in ['Impulse', 'ImpulseInteraction']:
            for response in self.responses():
                if response.name() == x.name():
                    out = True
                    break

        return out


    def add_rangf(self, rangf):
        """
        Add random grouping factor name to this interaction.

        :param rangf: ``str``; random grouping factor name
        :return: ``None``
        """

        if not isinstance(rangf, list):
            rangf = [rangf]
        for gf in rangf:
            if gf not in self.rangf:
                self.rangf.append(gf)

        self.rangf = sorted(self.rangf, key=lambda x: (x is None, x))

    def replace(self, old, new):
        """
        Replace an old input with a new one

        :param old: ``IRFNode``, ``Impulse``, and/or ``ImpulseInteraction`` object; response to remove.
        :param new: ``IRFNode``, ``Impulse``, and/or ``ImpulseInteraction`` object; response to add.
        :return: ``None``
        """

        ix = self.atomic_responses.index(old)
        self.atomic_responses[ix] = new

class IRFNode(object):
    """
    Data structure representing a node in a CDR IRF tree.
    For more information on how the CDR IRF structure is encoded as a tree, see the reference on CDR IRF trees.

    :param family: ``str``; name of IRF kernel family.
    :param impulse: ``Impulse`` object or ``None``; the impulse if terminal, else ``None``.
    :param p: ``IRFNode`` object or ``None``; the parent IRF node, or ``None`` if no parent (parent nodes can be connected after initialization).
    :param irfID: ``str`` or ``None``; string ID of node if applicable. If ``None``, automatically-generated ID will discribe node's family and structural position.
    :param coefID: ``str`` or ``None``; string ID of coefficient if applicable. If ``None``, automatically-generated ID will discribe node's family and structural position. Only applicable to terminal nodes, so this property will not be used if the node is non-terminal.
    :param ops: ``list`` of ``str``, or ``None``; ops to apply to IRF node. If ``None``, no ops.
    :param cont: ``bool``; Node connects directly to a continuous predictor. Only applicable to terminal nodes, so this property will not be used if the node is non-terminal.
    :param fixed: ``bool``; Whether node exists in the model's fixed effects structure.
    :param rangf: ``list`` of ``str``, ``str``, or ``None``; names of any random grouping factors associated with the node.
    :param param_init: ``dict``; map from parameter names to initial values, which will also be used as prior means.
    :param trainable: ``list`` of ``str``, or ``None``; trainable parameters at this node. If ``None``, all parameters are trainable.
    """

    def __init__(
            self,
            family=None,
            impulse=None,
            p=None,
            irfID=None,
            coefID=None,
            ops=None,
            cont=False,
            fixed=True,
            rangf=None,
            param_init=None,
            trainable=None
    ):
        if family is None or family in ['Terminal', 'DiracDelta']:
            assert irfID is None, 'Attempted to tie parameters (irf_id=%s) on parameter-free IRF node (family=%s)' % (irfID, family)
        if family != 'Terminal':
            assert coefID is None, 'Attempted to set coef_id=%s on non-terminal IRF node (family=%s)' % (coefID, family)
            assert impulse is None, 'Attempted to attach impulse (%s) to non-terminal IRF node (family=%s)' % (impulse, family)
            assert not cont, 'Attempted to set cont=True on non-terminal IRF node (family=%s)' % family
        if family is None:
            self.ops = []
            self.cont = False
            self.impulse = None
            self.family = None
            self.irfID = None
            self.coefID = None
            self.fixed = fixed
            self.rangf = []
            self.param_init={}
        else:
            self.ops = [] if ops is None else ops[:]
            self.cont = cont
            self.impulse = impulse
            self.family = family
            self.irfID = irfID
            self.coefID = coefID
            self.fixed = fixed
            self.rangf = [] if rangf is None else sorted(rangf) if isinstance(rangf, list) else [rangf]

            self.param_init = {}
            if param_init is not None:
                for param in Formula.irf_params(self.family):
                    if param in param_init:
                        self.param_init[param] = param_init[param]

        if trainable is None:
            self.trainable = Formula.irf_params(self.family)
        else:
            new_trainable = []
            for param in Formula.irf_params(self.family):
                if param in trainable:
                    new_trainable.append(param)
            self.trainable = new_trainable

        self.children = []
        self.p = p
        if self.p is not None:
            self.p.add_child(self)

        self.interaction_list = []

    def add_child(self, t):
        """
        Add child to this node in the IRF tree

        :param t: ``IRFNode``; child node.
        :return: ``IRFNode``; child node with updated parent.
        """

        if self.terminal():
            raise ValueError('Tried to add child to terminal IRFNode')
        child_names = [c.local_name() for c in self.children]
        if t.local_name() in child_names:
            c = self.children[child_names.index(t.local_name())]
            c.add_rangf(t.rangf)
            for c_t in t.children:
                c.add_child(c_t)
            out = c
        else:
            self.children.append(t)
            t.p = self
            out = t

        return out

    def add_rangf(self, rangf):
        """
        Add random grouping factor name to this node.

        :param rangf: ``str``; random grouping factor name
        :return: ``None``
        """

        if not isinstance(rangf, list):
            rangf = [rangf]
        if len(rangf) == 0 and self.terminal():
            self.fixed = True
        for gf in rangf:
            if gf not in self.rangf:
                self.rangf.append(gf)

        self.rangf = sorted(self.rangf)

    def add_interactions(self, response_interactions):
        """
        Add a ResponseInteraction object (or list of them) to this node.

        :param response_interaction: ``ResponseInteraction`` or ``list`` of ``ResponseInteraction``; response interaction(s) to add
        :return: ``None``
        """

        assert self.terminal() or response_interactions is None or len(response_interactions) == 0, 'Interactions cannot be added to non-terminal IRF nodes.'

        if not isinstance(response_interactions, list):
            response_interactions = [response_interactions]

        new_interactions = []
        cur_interaction_names = [x.name() for x in self.interaction_list]
        for r in response_interactions:
            assert type(r).__name__ == 'ResponseInteraction', 'All inputs to add_interactions() must be of type ResponseInteraction. Got type %s.' % type(r).__name__
            if r.name() in cur_interaction_names:
                ix = cur_interaction_names.index(r.name())
                old_interaction = self.interaction_list[ix]
                for gf in r.rangf:
                    old_interaction.add_rangf(gf)
            else:
                new_interactions.append(r)

        interaction_list = self.interaction_list + new_interactions
        self.interaction_list = sorted(list(set(interaction_list)), key = lambda x: x.name())

    def interactions(self):
        """
        Return list of all response interactions used in this subtree, sorted by name.

        :return: ``list`` of ``ResponseInteraction``
        """

        interaction_list = []

        if self.terminal():
            interaction_list += self.interaction_list
        else:
            for c in self.children:
                interaction_list += c.interactions()

        return sorted(list(set(interaction_list)), key = lambda x: x.name())

    def interaction_names(self):
        """
        Get list of names of interactions dominated by node.

        :return: ``list`` of ``str``; names of interactions dominated by node.
        """

        if self.terminal():
            out = [x.name() for x in self.interactions()]
        else:
            out = []
            for c in self.children:
                names = c.interaction_names()
                for name in names:
                    out.append(name)

        out = sorted(list(set(out)))

        return out

    def fixed_interaction_names(self):
        """
        Get list of names of fixed interactions dominated by node.

        :return: ``list`` of ``str``; names of fixed interactions dominated by node.
        """

        if self.terminal():
            out = [x.name() for x in self.interactions() if None in x.rangf]
        else:
            out = []
            for c in self.children:
                names = c.fixed_interaction_names()
                for name in names:
                    out.append(name)

        out = sorted(list(set(out)))

        return out

    def interactions2inputs(self):
        """
        Get map from IDs of ResponseInteractions dominated by node to lists of IDs of their inputs.

        :return: ``dict``; map from IDs of ResponseInteractions nodes to lists of their inputs.
        """

        out = {}
        for x in self.interactions():
            interaction = x.name()
            out[interaction] = [y.name() for y in x.responses()]

        return out


    def local_name(self):
        """
        Get descriptive name for this node, ignoring its position in the IRF tree.

        :return: ``str``; name.
        """

        if self.irfID is None:
            out = '.'.join([self.family] + self.impulse_names())
        else:
            out = self.irfID
        return out

    def name(self):
        """
        Get descriptive name for this node.

        :return: ``str``; name.
        """

        if self.family is None:
            return 'ROOT'
        if self.p is None or self.p.name() == 'ROOT':
            p_name = ''
        else:
            p_name = self.p.name() + '-'
        return p_name + self.local_name()

    def irf_to_formula(self, rangf=None):
        """
        Generates a representation of this node's impulse response kernel in formula string syntax

        :param rangf: random grouping factor for which to generate the stringification (fixed effects if rangf==None).

        :return: ``str``; formula string representation of node
        """

        if self.family is None:
            out = ''
        else:
            params = []
            if self.irfID is not None:
                params.append('irf_id=%s' % self.irfID)
            for c in self.children:
                if c.terminal() and c.coefID is not None:
                    params.append('coef_id=%s' % c.coefID)
                    break
            if rangf in self.rangf:
                params.append('ran=T')
            for c in self.children:
                if c.terminal() and c.cont:
                    params.append('cont=T')
            if len(self.param_init) > 0:
                params.append(', '.join(['%s=%s' % (x, self.param_init[x]) for x in self.param_init]))
            if set(self.trainable) != set(Formula.irf_params(self.family)):
                params.append('trainable=%s' % self.trainable)

            if self.p is not None:
                inner = self.p.irf_to_formula(rangf)
            else:
                inner = ''

            if self.family != 'Terminal':
                if inner != '' and len(params) > 0:
                    inner = inner + ', ' + ', '.join(params)
                elif len(params) > 0:
                    inner = ', '.join(params)
                out = self.family + '(' + inner + ')'
            else:
                out = inner

        return out

    def irf_id(self):
        """
        Get IRF ID for this node.

        :return: ``str`` or ``None``; IRF ID, or ``None`` if terminal.
        """

        if not self.terminal():
            if self.irfID is None:
                out = self.name()
            else:
                out = self.irfID
            return out
        return None

    def coef_id(self):
        """
        Get coefficient ID for this node.

        :return: ``str`` or ``None``; coefficient ID, or ``None`` if non-terminal.
        """
        if self.terminal():
            if self.coefID is None:
                return self.name()
            return self.coefID
        return None

    def terminal(self):
        """
        Check whether node is terminal.

        :return: ``bool``; whether node is terminal.
        """

        return self.family == 'Terminal'

    def depth(self):
        """
        Get depth of node in tree.

        :return: ``int``; depth
        """

        d = 1
        for c in self.children:
            if c.depth() + 1 > d:
                d = c.depth() + 1
        return d

    def has_composed_irf(self):
        """
        Check whether node dominates any IRF compositions.

        :return: ``bool``, whether node dominates any IRF compositions.
        """

        return self.depth() > 3

    def is_LCG(self):
        """
        Check the non-parametric type of a node's kernel, or return ``None`` if parametric.

        :param family: ``str``; name of IRF family
        :return: ``str`` or ``None; name of kernel type if non-parametric, else ``None``.
        """
        return Formula.is_LCG(self.family)

    def bases(self):
        """
        Get the number of bases of node.

        :return: ``int`` or ``None``; number of bases of node, or ``None`` if node is not a spline.
        """
        return Formula.bases(self.family)

    def impulses(self, include_interactions=False):
        """
        Get list of impulses dominated by node.
    
        :param include_interactions: ``bool``; whether to return impulses defined by interaction terms.

        :return: ``list`` of ``Impulse``; impulses dominated by node.
        """

        out = []
        if self.terminal():
            out.append(self.impulse)
            if include_interactions:
                for interaction in self.interactions():
                    for response in interaction.responses():
                        if isinstance(response, IRFNode):
                            if response.impulse.name() not in [x.name() for x in out]:
                                out.append(response.impulse)
                        elif isinstance(response, ImpulseInteraction):
                            for subresponse in response.impulses():
                                if subresponse.name() not in [x.name() for x in out]:
                                    out.append(subresponse)
                        elif isinstance(response, Impulse):
                            if response.name() not in [x.name() for x in out]:
                                out.append(response)
                        else:
                            raise ValueError('Unsupported type "%s" for input to interaction' % type(response).__name__)
        else:
            for c in self.children:
                for imp in c.impulses(include_interactions=include_interactions):
                    if imp.name() not in [x.name() for x in out]:
                        out.append(imp)
        return out

    def impulses_from_response_interaction(self):
        """
        Get list of any impulses from response interactions associated with this node.

        :return: ``list`` of ``Impulse``; impulses dominated by node.
        """

        out = []
        if self.terminal():
            out.append(self.impulse)
            for interaction in self.interactions():
                for response in interaction.responses():
                    if not isinstance(response, IRFNode):
                        if response.name() not in [x.name() for x in out]:
                            out.append(response)
        else:
            for c in self.children:
                for imp in c.impulses_from_response_interaction():
                    if imp.name() not in [x.name() for x in out]:
                        out.append(imp)

        return out

    def impulse_names(self, include_interactions=False):
        """
        Get list of names of impulses dominated by node.

        :param include_interactions: ``bool``; whether to return impulses defined by interaction terms.
       
        :return: ``list`` of ``str``; names of impulses dominated by node.
        """

        return [x.name() for x in self.impulses(include_interactions=include_interactions)]

    def impulses_by_name(self):
        """
        Get dictionary mapping names of impulses dominated by node to their corresponding impulses.

        :return: ``dict``; map from impulse names to impulses
        """

        out = {}
        for x in self.impulses():
            out[x.name()] = x

        return out

    def terminals(self):
        """
        Get list of terminal IRF nodes dominated by node.

        :return: ``list`` of ``IRFNode``; terminal IRF nodes dominated by node.
        """

        out = []
        if self.terminal():
            out.append(self)
        else:
            for c in self.children:
                for term in c.terminals():
                    if term.name() not in [x.name() for x in out]:
                        out.append(term)
        return out

    def terminal_names(self):
        """
        Get list of names of terminal IRF nodes dominated by node.

        :return: ``list`` of ``str``; names of terminal IRF nodes dominated by node.
        """

        return [x.name() for x in self.terminals()]

    def terminals_by_name(self):
        """
        Get dictionary mapping names of terminal IRF nodes dominated by node to their corresponding nodes.

        :return: ``dict``; map from node names to nodes
        """

        out = {}
        for x in self.terminals():
            out[x.name()] = x
        return out

    def coef_names(self):
        """
        Get list of names of coefficients dominated by node.

        :return: ``list`` of ``str``; names of coefficients dominated by node.
        """

        out = []
        if self.terminal():
            out.append(self.coef_id())
        else:
            for c in self.children:
                names = c.coef_names()
                for name in names:
                    if name not in out:
                        out.append(name)
        return out

    def fixed_coef_names(self):
        """
        Get list of names of fixed coefficients dominated by node.

        :return: ``list`` of ``str``; names of fixed coefficients dominated by node.
        """

        out = []
        if self.terminal():
            child_coefs = set()
            for c in self.p.children:
                child_coefs.add(c.coef_id())
            if self.fixed:
                out.append(self.coef_id())
        else:
            for c in self.children:
                names = c.fixed_coef_names()
                for name in names:
                    if name not in out:
                        out.append(name)
        return out

    def nonparametric_coef_names(self):
        """
        Get list of names of nonparametric coefficients dominated by node.
        :return: ``list`` of ``str``; names of spline coefficients dominated by node.
        """

        out = []
        if self.terminal():
            if Formula.is_LCG(self.p.family):
                out.append(self.coef_id())
        else:
            for c in self.children:
                names = c.nonparametric_coef_names()
                for name in names:
                    if name not in out:
                        out.append(name)
        return out

    def unary_nonparametric_coef_names(self):
        """
        Get list of names of non-parametric coefficients with no siblings dominated by node.
        Because unary splines are non-parametric, their coefficients are fixed at 1.
        Trainable coefficients are therefore perfectly confounded with the spline parameters.
        Splines dominating multiple coefficients are excepted, since the same kernel shape must be scaled in different ways.

        :return: ``list`` of ``str``; names of unary spline coefficients dominated by node.
        """

        out = []
        if self.terminal():
            if Formula.is_LCG(self.p.family):
                child_coefs = set()
                for c in self.p.children:
                    child_coefs.add(c.coef_id())
                if len(child_coefs) == 1:
                    out.append(self.coef_id())
        else:
            for c in self.children:
                names = c.unary_nonparametric_coef_names()
                for name in names:
                    if name not in out:
                        out.append(name)
        return out

    def atomic_irf_by_family(self):
        """
        Get map from IRF kernel family names to list of IDs of IRFNode instances belonging to that family.

        :return: ``dict`` from ``str`` to ``list`` of ``str``; IRF IDs by family.
        """

        if self.family is None or self.family == 'Terminal':
            out = {}
        else:
            out = {self.family: [self.irf_id()]}
        for c in self.children:
            c_id_by_family = c.atomic_irf_by_family()
            for f in c_id_by_family:
                if f not in out:
                    out[f] = c_id_by_family[f]
                else:
                    for irf in c_id_by_family[f]:
                        if irf not in out[f]:
                            out[f].append(irf)
        return out

    def atomic_irf_param_init_by_family(self):
        """
        Get map from IRF kernel family names to maps from IRF IDs to maps from IRF parameter names to their initialization values.

        :return: ``dict``; parameter initialization maps by family.
        """

        if self.family is None or self.family == 'Terminal':
            out = {}
        else:
            out = {self.family: {self.irf_id(): self.param_init}}
        for c in self.children:
            c_id_by_family = c.atomic_irf_param_init_by_family()
            for f in c_id_by_family:
                if f not in out:
                    out[f] = c_id_by_family[f]
                else:
                    for irf in c_id_by_family[f]:
                        if irf not in out[f]:
                            out[f][irf] = c_id_by_family[f][irf]
        return out

    def atomic_irf_param_trainable_by_family(self):
        """
        Get map from IRF kernel family names to maps from IRF IDs to lists of trainable parameters.

        :return: ``dict``; trainable parameter maps by family.
        """
        if self.family is None or self.family == 'Terminal':
            out = {}
        else:
            out = {self.family: {self.irf_id(): self.trainable}}
        for c in self.children:
            c_id_by_family = c.atomic_irf_param_trainable_by_family()
            for f in c_id_by_family:
                if f not in out:
                    out[f] = c_id_by_family[f]
                else:
                    for irf in c_id_by_family[f]:
                        if irf not in out[f]:
                            out[f][irf] = c_id_by_family[f][irf]
        return out

    def supports_non_causal(self):
        """
        Check whether model contains only IRF kernels that lack the causality constraint t >= 0.

        :return: ``bool``: whether model contains only IRF kernels that lack the causality constraint t >= 0.
        """

        out = self.family not in Formula.CAUSAL_IRFS
        if out:
            for c in self.children:
                out &= c.supports_non_causal()
                if not out:
                    break

        return out

    def has_coefficient(self, rangf):
        """
        Report whether **rangf** has any coefficients in this subtree

        :param rangf: Random grouping factor
        :return: ``bool``: Whether **rangf** has any coefficients in this subtree
        """

        if self.terminal() and rangf in self.rangf:
            return True
        else:
            for c in self.children:
                if c.has_coefficient(rangf):
                    return True
        return False

    def has_irf(self, rangf):
        """
        Report whether **rangf** has any IRFs in this subtree

        :param rangf: Random grouping factor
        :return: ``bool``: Whether **rangf** has any IRFs in this subtree
        """

        if not self.terminal() and not self.family == 'DiracDelta' and rangf in self.rangf:
            return True
        else:
            for c in self.children:
                if c.has_irf(rangf):
                    return True
        return False

    def coef2impulse(self):
        """
        Get map from coefficient IDs dominated by node to lists of corresponding impulses.

        :return: ``dict``; map from coefficient IDs to lists of corresponding impulses.
        """

        out = {}
        for x in self.terminals():
            coef = x.coef_id()
            imp = x.impulse.name()
            if coef not in out:
                out[coef] = [imp]
            else:
                if imp not in out[coef]:
                    out[coef].append(imp)
        return out

    def impulse2coef(self):
        """
        Get map from impulses dominated by node to lists of corresponding coefficient IDs.

        :return: ``dict``; map from impulses to lists of corresponding coefficient IDs.
        """

        out = {}
        for x in self.terminals():
            coef = x.coef_id()
            imp = x.impulse.name()
            if imp not in out:
                out[imp] = [coef]
            else:
                if coef not in out[imp]:
                    out[imp].append(coef)
        return out

    def coef2terminal(self):
        """
        Get map from coefficient IDs dominated by node to lists of corresponding terminal IRF nodes.

        :return: ``dict``; map from coefficient IDs to lists of corresponding terminal IRF nodes.
        """

        out = {}
        for x in self.terminals():
            coef = x.coef_id()
            term = x.name()
            if coef not in out:
                out[coef] = [term]
            else:
                if term not in out[coef]:
                    out[coef].append(term)
        return out

    def terminal2coef(self):
        """
        Get map from IDs of terminal IRF nodes dominated by node to lists of corresponding coefficient IDs.

        :return: ``dict``; map from IDs of terminal IRF nodes to lists of corresponding coefficient IDs.
        """

        out = {}
        for x in self.terminals():
            coef = x.coef_id()
            term = x.name()
            if term not in out:
                out[term] = [coef]
            else:
                if coef not in out[term]:
                    out[term].append(coef)
        return out

    def terminal2impulse(self):
        """
        Get map from terminal IRF nodes dominated by node to lists of corresponding impulses.

        :return: ``dict``; map from terminal IRF nodes to lists of corresponding impulses.
        """

        out = {}
        for x in self.terminals():
            term = x.name()
            imp = x.impulse.name()
            if term not in out:
                out[term] = [imp]
            else:
                if imp not in out[term]:
                    out[term].append(imp)
        return out

    def impulse2terminal(self):
        """
        Get map from impulses dominated by node to lists of corresponding terminal IRF nodes.

        :return: ``dict``; map from impulses to lists of corresponding terminal IRF nodes.
        """

        out = {}
        for x in self.terminals():
            term = x.name()
            imp = x.impulse.name()
            if imp not in out:
                out[imp] = [term]
            else:
                if term not in out[imp]:
                    out[imp].append(term)
        return out

    def coef_by_rangf(self):
        """
        Get map from random grouping factor names to associated coefficient IDs dominated by node.

        :return: ``dict``; map from random grouping factor names to associated coefficient IDs.
        """

        out = {}
        if self.terminal():
            for gf in self.rangf:
                child_coefs = set()
                for c in self.p.children:
                    child_coefs.add(c.coef_id())
                out[gf] = []
                if self.coef_id() not in out[gf]:
                    out[gf].append(self.coef_id())
        for c in self.children:
            c_out = c.coef_by_rangf()
            for gf in c_out:
                for x in c_out[gf]:
                    if gf not in out:
                        out[gf] = []
                    if x not in out[gf]:
                        out[gf].append(x)
        return out

    def interaction_by_rangf(self):
        """
        Get map from random grouping factor names to associated interaction IDs dominated by node.

        :return: ``dict``; map from random grouping factor names to associated interaction IDs.
        """

        out = {}
        if self.terminal():
            for interaction in self.interactions():
                for gf in interaction.rangf:
                    if gf is not None:
                        if not gf in out:
                            out[gf] = set()
                        if not interaction.name() in out[gf]:
                            out[gf].add(interaction.name())
        for c in self.children:
            c_out = c.interaction_by_rangf()
            for gf in c_out:
                for x in c_out[gf]:
                    if gf not in out:
                        out[gf] = set()
                    if x not in out[gf]:
                        out[gf].add(x)
        for gf in out:
            out[gf] = sorted(list(out[gf]))

        return out

    def irf_by_rangf(self):
        """
        Get map from random grouping factor names to IDs of associated IRF nodes dominated by node.

        :return: ``dict``; map from random grouping factor names to IDs of associated IRF nodes.
        """

        out = {}
        if not self.terminal():
            for gf in self.rangf:
                out[gf] = []
                if self.irf_id() not in out[gf]:
                    out[gf].append(self.irf_id())
        for c in self.children:
            c_out = c.irf_by_rangf()
            for gf in c_out:
                for x in c_out[gf]:
                    if gf not in out:
                        out[gf] = []
                    if x not in out[gf]:
                        out[gf].append(x)
        return out

    def node_table(self):
        """
        Get map from names to nodes of all nodes dominated by node (including self).

        :return: ``dict``; map from names to nodes of all nodes dominated by node.
        """

        out = {self.name(): self}
        for c in self.children:
            nt = c.node_table()
            for x in nt:
                assert not x in out, 'Duplicate node ids appear in IRF tree'
                out[x] = nt[x]
        return out

    def pc_transform(self, n_pc, pointers=None):
        """
        Generate principal-components-transformed copy of node.
        Recursive.
        Returns a tree forest representing the current state of the transform.
        When run from ROOT, should always return a length-1 list representing a single-tree forest, in which case the transformed tree is accessible as the 0th element.

        :param n_pc: ``int``; number of principal components in transform.
        :param pointers: ``dict``; map from source nodes to transformed nodes.
        :return: ``list`` of ``IRFNode``; tree forest representing current state of the transform.
        """

        self_transformed = []

        if self.terminal():
            if self.impulse.name() == 'rate':
                self_pc = IRFNode(
                    family='Terminal',
                    impulse=self.impulse,
                    coefID=self.coefID,
                    cont=self.cont,
                    fixed=self.fixed,
                    rangf=self.rangf[:],
                    param_init=self.param_init,
                    trainable=self.trainable
                )
                self_transformed.append(self_pc)
                if pointers is not None:
                    if self not in pointers:
                        pointers[self] = []
                    pointers[self].append(self_pc)
            else:
                pc_impulses = [Impulse('%d' % i) for i in range(n_pc)]
                for x in pc_impulses:
                    self_pc = IRFNode(
                        family='Terminal',
                        impulse=x,
                        coefID=self.coefID,
                        cont=self.cont,
                        fixed=self.fixed,
                        rangf=self.rangf[:],
                        param_init=self.param_init,
                        trainable=self.trainable
                    )
                    self_transformed.append(self_pc)
                    if pointers is not None:
                        if self not in pointers:
                            pointers[self] = []
                        pointers[self].append(self_pc)

        elif self.family is None:
            ## ROOT node
            children = []
            for c in self.children:
                c_children = [x for x in c.pc_transform(n_pc, pointers)]
                children += c_children
            self_pc = IRFNode()
            for c in children:
                c_new = self_pc.add_child(c)
                if c_new != c:
                    if pointers is not None:
                        if c in pointers:
                            pointers[c_new] = pointers[c]
                            del pointers[c]
            self_transformed.append(self_pc)
            if pointers is not None:
                if self not in pointers:
                    pointers[self] = []
                pointers[self].append(self_pc)

        else:
            children = []
            for c in self.children:
                c_children = [x for x in c.pc_transform(n_pc, pointers)]
                children += c_children
            for c in children:
                self_pc = IRFNode(
                    family=self.family,
                    irfID=self.irfID,
                    fixed=self.fixed,
                    rangf=self.rangf,
                    param_init=self.param_init,
                    trainable=self.trainable
                )
                c_new = self_pc.add_child(c)
                if c_new != c:
                    if pointers is not None:
                        if c in pointers:
                            pointers[c_new] = pointers[c]
                            del pointers[c]
                self_transformed.append(self_pc)
                if pointers is not None:
                    if self not in pointers:
                        pointers[self] = []
                    pointers[self].append(self_pc)
        return self_transformed

    def categorical_transform(self, X, expansion_map=None):
        """
        Generate transformed copy of node with categorical predictors in **X** expanded.
        Recursive.
        Returns a tree forest representing the current state of the transform.
        When run from ROOT, should always return a length-1 list representing a single-tree forest, in which case the transformed tree is accessible as the 0th element.

        :param X: list of ``pandas`` tables; input data.
        :param expansion_map: ``dict``; Internal variable. Do not use.
        :return: ``list`` of ``IRFNode``; tree forest representing current state of the transform.
        """

        if expansion_map is None:
            top = True
            expansion_map = {}
        else:
            top = False

        self_transformed = []

        if self.terminal():
            for interaction in self.interactions():
                for response in interaction.responses():
                    if isinstance(response, Impulse):
                        if not response.name() in expansion_map:
                            if response.categorical(X):
                                found = False
                                for _X in X:
                                    if response.id in _X:
                                        found = True
                                        vals = sorted(_X[response.id].unique()[1:])
                                        expansion = [Impulse('_'.join([response.id, pythonize_string(str(val))]), ops=response.ops) for val in vals]
                                        break
                                assert found, 'Impulse %d not found in data.' % response.id
                            else:
                                expansion = [response]
                            expansion_map[response.name()] = expansion
                    elif isinstance(response, ImpulseInteraction):
                        for subresponse in response.impulses():
                            if not subresponse.name() in expansion_map:
                                if subresponse.categorical(X):
                                    found = False
                                    for _X in X:
                                        if subresponse.id in _X:
                                            found = True
                                            vals = sorted(_X[subresponse.id].unique()[1:])
                                            expansion = [
                                                Impulse('_'.join([subresponse.id, pythonize_string(str(val))]), ops=subresponse.ops)
                                                for val in vals]
                                            break
                                    assert found, 'Impulse %d not found in data.' % subresponse.id
                                else:
                                    expansion = [subresponse]
                                expansion_map[subresponse.name()] = expansion

            if type(self.impulse).__name__ == 'ImpulseInteraction':
                expanded_atomic_impulses = []
                for x in self.impulse.impulses():
                    if x.name() not in expansion_map:
                        if x.categorical(X):
                            found = False
                            for _X in X:
                                if x.id in _X:
                                    found = True
                                    vals = sorted(_X[x.id].unique()[1:])
                                    expansion = [Impulse('_'.join([x.id, pythonize_string(str(val))]), ops=x.ops) for val in vals]
                                    break
                            assert found, 'Impulse %s not found in data.' % x.id
                        else:
                            expansion = [x]
                        expansion_map[x.name()] = expansion

                    expanded_atomic_impulses.append(expansion_map[x.name()])

                new_impulses = [ImpulseInteraction(x, ops=self.impulse.ops) for x in itertools.product(*expanded_atomic_impulses)]

            else:
                if not self.impulse.name() in expansion_map:
                    if self.impulse.categorical(X):
                        if self.impulse.categorical(X):
                            found = False
                            for _X in X:
                                if self.impulse.id in _X:
                                    found = True
                                    vals = sorted(_X[self.impulse.id].unique()[1:])
                                    expansion = [Impulse('_'.join([self.impulse.id, pythonize_string(str(val))]), ops=self.impulse.ops) for val in vals]
                                    break
                            assert found, 'Impulse %s not found in data.' % self.impulse.id
                        else:
                            expansion = [self.impulse]
                        expansion_map[self.impulse.name()] = expansion
                    else:
                        expansion_map[self.impulse.name()] = [self.impulse]

                new_impulses = expansion_map[self.impulse.name()]

            irf_expansion = []

            for x in new_impulses:
                new_irf = IRFNode(
                    family='Terminal',
                    impulse=x,
                    coefID=self.coefID,
                    cont=self.cont,
                    fixed=self.fixed,
                    rangf=self.rangf[:],
                    param_init=self.param_init,
                    trainable=self.trainable
                )
                irf_expansion.append(new_irf)

                self_transformed.append(new_irf)

            expansion_map[self.name()] = irf_expansion

        elif self.family is None:
            ## ROOT node
            children = []
            for c in self.children:
                c_children = [x for x in c.categorical_transform(X, expansion_map=expansion_map)]
                children += c_children
            new_irf = IRFNode()
            for c in children:
                new_irf.add_child(c)
            self_transformed.append(new_irf)

        else:
            children = []
            for c in self.children:
                c_children = [x for x in c.categorical_transform(X, expansion_map=expansion_map)]
                children += c_children
            for c in children:
                new_irf = IRFNode(
                    family=self.family,
                    irfID=self.irfID,
                    fixed=self.fixed,
                    rangf=self.rangf,
                    param_init=self.param_init,
                    trainable=self.trainable
                )
                new_irf.add_child(c)
                self_transformed.append(new_irf)

        if top:
            old_interactions = self.interactions()
            new_interactions = []

            for old_interaction in old_interactions:
                old_rangf = old_interaction.rangf[:]
                expanded_interaction = []
                for response in old_interaction.responses():
                    if isinstance(response, ImpulseInteraction):
                        expanded_interaction_impulse = []
                        for impulse in response.impulses():
                            expansion = expansion_map[impulse.name()]
                            expanded_interaction_impulse.append(expansion)
                        expanded_interaction_impulse = [ImpulseInteraction(x, ops=response.ops) for x in itertools.product(*expanded_interaction_impulse)]
                        expanded_interaction.append(expanded_interaction_impulse)
                    else:
                        expansion = expansion_map[response.name()]
                        expanded_interaction.append(expansion)
                expanded_interaction = [ResponseInteraction(x, rangf=old_rangf) for x in itertools.product(*expanded_interaction)]
                new_interactions += expanded_interaction
            for interaction in new_interactions:
                for irf in interaction.irf_responses():
                    irf.add_interactions(interaction)

        return self_transformed

    @staticmethod
    def pointers2namemmaps(p):
        """
        Get a map from source to transformed IRF node names.

        :param p: ``dict``; map from source to transformed IRF nodes.
        :return: ``dict``; map from source to transformed IRF node names.
        """

        fw = {}
        bw = {}
        for t in p:
            for t_pc in p[t]:
                if not t.name() in fw:
                    fw[t.name()] = []
                if not t_pc.name() in fw[t.name()]:
                    fw[t.name()].append(t_pc.name())
                if not t_pc.name() in bw:
                    bw[t_pc.name()] = []
                if not t_pc.name() in bw[t_pc.name()]:
                    bw[t_pc.name()].append(t.name())
        return fw, bw

    def ablate_impulses(self, impulse_ids):
        """
        Remove impulses in **impulse_ids** from fixed effects (retaining in any random effects).

        :param impulse_ids: ``list`` of ``str``; impulse ID's
        :return: ``None``
        """

        if not isinstance(impulse_ids, list):
            impulse_ids = [impulse_ids]
        if self.terminal():
            if self.impulse.id in impulse_ids:
                self.fixed = False
            else:
                for interaction in self.interactions():
                    if None in interaction.rangf:
                        ids = []
                        for response in interaction.responses():
                            if isinstance(response, IRFNode):
                                ids.append(response.impulse.id)
                            elif isinstance(response, ImpulseInteraction):
                                for subresponse in response.impulses():
                                    ids.append(subresponse.impulse.id)
                            else:
                                ids.append(response.id)
                        ids = ':'.join(ids)
                        if ids in impulse_ids:
                            interaction.rangf.remove(None)
        else:
            for c in self.children:
                c.ablate_impulses(impulse_ids)

    def unablate_impulses(self, impulse_ids):
        """
        Insert impulses in **impulse_ids** into fixed effects (leaving random effects structure unchanged).

        :param impulse_ids: ``list`` of ``str``; impulse ID's
        :return: ``None``
        """

        if not isinstance(impulse_ids, list):
            impulse_ids = [impulse_ids]
        if self.terminal():
            if self.impulse.id in impulse_ids:
                self.fixed = True
        else:
            for c in self.children:
                c.unablate_impulses(impulse_ids)

    def remove_impulses(self, impulse_ids):
        """
        Remove impulses in **impulse_ids** from the model (both fixed and random effects).

        :param impulse_ids: ``list`` of ``str``; impulse ID's
        :return:  ``None``
        """

        if not isinstance(impulse_ids, list):
            impulse_ids = [impulse_ids]
        if not self.terminal():
            new_children = []
            for c in self.children:
                if not (c.terminal() and c.impulse.id in impulse_ids):
                    c.remove_impulses(impulse_ids)
                    if c.terminal() or len(c.children) > 0:
                        new_children.append(c)
            self.children = new_children

    def formula_terms(self):
        """
        Return data structure representing formula terms dominated by node, grouped by random grouping factor.
        Key ``None`` represents the fixed portion of the model (no random grouping factor).

        :return: ``dict``; map from random grouping factors data structure representing formula terms.
            Data structure contains 2 fields, ``'impulses'`` containing impulses and ``'irf'`` containing IRF Nodes.
        """

        if self.terminal():
            out = {}
            if self.fixed:
                out[None] = [{
                    'impulses': self.impulses(),
                    'irf': ''
                }]
            for rangf in self.rangf:
                out[rangf] = [{
                    'impulses': self.impulses(),
                    'irf': ''
                }]

            return out

        out = {}
        for c in self.children:
            c_data = c.formula_terms()
            for key in c_data:
                if key in out:
                    out[key] += c_data[key]
                else:
                    out[key] = c_data[key]

        if self.family is not None:
            for key in out:
                for term in out[key]:
                    term['irf'] = self.irf_to_formula(rangf=key)

        return out

    def __str__(self):
        s = self.name()
        if len(self.rangf) > 0:
            s += '; rangf: ' + ','.join(self.rangf)
        if len(self.trainable) > 0:
            s +=  '; trainable params: ' + ', '.join(self.trainable)
        indent = '  '
        for c in self.children:
            s += '\n%s' % indent + str(c).replace('\n', '\n%s' % indent)
        return s

