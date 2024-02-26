import sys
import re
import math
import ast
import itertools
import numpy as np

from .data import z, c, s, compute_time_mask, expand_impulse_sequence
from .kwargs import NN_KWARGS, NN_BAYES_KWARGS
from .util import names2ix, sn, stderr

sys.setrecursionlimit(100000)  # This prevents stackoverflow for formulas with lots of terms

interact = re.compile('([^ ]+):([^ ]+)')
spillover = re.compile('(^.+)S([0-9]+)$')
split_irf = re.compile('(.+)\(([^(]+)')
lcg_re = re.compile('(G|S|LCG)(b([0-9]+))?$')
starts_numeric = re.compile('^[0-9]')
non_alphanumeric = re.compile('[^0-9a-zA-Z_]')

NN_KWARG_MAP = {}
for x in NN_KWARGS + NN_BAYES_KWARGS:
    NN_KWARG_MAP[x.key] = [x.key]
    for a in x.aliases:
        if a in NN_KWARG_MAP:
            NN_KWARG_MAP[a] += [x.key]
        else:
            NN_KWARG_MAP[a] = [x.key]


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


def get_bool(k):
    is_false = ('False', 'FALSE', 'false', 'F', '0', 0, None, False)
    if type(k.value).__name__ == 'Constant':
        if isinstance(k.value.value, str):
            if k.value.s in is_false:
                out = False
            else:
                out = True
        elif k.value.n == 0:
            out = False
        else:
            out = True
    elif type(k.value).__name__ == 'Str':
        if k.value.s in is_false:
            out = False
        else:
            out = True
    elif type(k.value).__name__ == 'Name':
        if k.value.id in is_false:
            out = False
        else:
            out = True
    elif type(k.value).__name__ == 'NameConstant':
        if k.value.value in is_false:
            out = False
        else:
            out = True
    elif type(k.value).__name__ == 'Num':
        if k.value.n == 0:
            out = False
        else:
            out = True
    else:
        out = None

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
        'Normal': ['mu', 'sigma'],
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
        'NN': [],
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

    IRF_ALIASES = {
        'HRF1': 'HRFDoubleGamma1',
        'HRF2': 'HRFDoubleGamma2',
        'HRF3': 'HRFDoubleGamma3',
        'HRF4': 'HRFDoubleGamma4',
        'HRF5': 'HRFDoubleGamma5',
        'HRF': 'HRFDoubleGamma5',
    }

    RESPONSE_DISTRIBUTIONS = {
        'normal': {
            'dist': 'Normal',
            'name': 'normal',
            'params': ('mu', 'sigma'),
            'params_tf': ('loc', 'scale'),
            'support': 'real'
        },
        'lognormal': {
            'dist': 'LogNormal',
            'name': 'lognormal',
            'params': ('mu', 'sigma'),
            'params_tf': ('loc', 'scale'),
            'support': 'real'
        },
        'lognormalv2': {
            'dist': 'LogNormalV2',
            'name': 'lognormalv2',
            'params': ('mu', 'sigma'),
            'params_tf': ('loc', 'scale'),
            'support': 'real'
        },
        'sinharcsinh': {
            'dist': 'SinhArcsinh',
            'name': 'sinharcsinh',
            'params': ('mu', 'sigma', 'skewness', 'tailweight'),
            'params_tf': ('loc', 'scale', 'skewness', 'tailweight'),
            'support': 'real'
        },
        'johnsonsu': {
            'dist': 'JohnsonSU',
            'name': 'johnsonsu',
            'params': ('mu', 'sigma', 'skewness', 'tailweight'),
            'params_tf': ('loc', 'scale', 'skewness', 'tailweight'),
            'support': 'real'
        },
        'bernoulli': {
            'dist': 'Bernoulli',
            'name': 'bernoulli',
            'params': ('logit',),
            'params_tf': ('logits',),
            'support': 'discrete'
        },
        'categorical': {
            'dist': 'Categorical',
            'name': 'categorical',
            'params': ('logit',),
            'params_tf': ('logits',),
            'support': 'discrete'
        },
        'exponential': {
            'dist': 'Exponential',
            'name': 'exponential',
            'params': ('beta'),
            'params_tf': ('rate',),
            'support': 'positive'
        },
        'exgaussian': {
            'dist': 'ExponentiallyModifiedGaussian',
            'name': 'exgaussian',
            'params': ('mu', 'sigma', 'beta'),
            'params_tf': ('loc', 'scale', 'rate',),
            'support': 'real'
        },
        'exgaussianinvrate': {
            'dist': 'ExponentiallyModifiedGaussianInvRate',
            'name': 'exgaussianinvrate',
            'params': ('mu', 'sigma', 'beta'),
            'params_tf': ('loc', 'scale', 'rate',),
            'support': 'real'
        }
    }

    LCG_BASES_IX = 3
    LCG_DEFAULT_BASES = 10

    @staticmethod
    def normalize_irf_family(family):
        return Formula.IRF_ALIASES.get(family, family)

    @staticmethod
    def irf_params(family):
        """
        Return list of parameter names for a given IRF family.

        :param family: ``str``; name of IRF family
        :return: ``list`` of ``str``; parameter names
        """

        family = Formula.normalize_irf_family(family)

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

        family = Formula.normalize_irf_family(family)

        return family is not None and lcg_re.match(family) is not None

    @staticmethod
    def bases(family):
        """
        Get the number of bases of a spline kernel.

        :param family: ``str``; name of IRF family
        :return: ``int`` or ``None``; number of bases of spline kernel, or ``None`` if **family** is not a spline.
        """

        family = Formula.normalize_irf_family(family)

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
        out = s.strip().replace('.(', '(').replace(':', '%').replace('^', '**')
        out = re.sub(r'pow([0-9]*)\.([0-9])', r'pow\1_\2', out)
        return out

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
        terms = []
        if len(rhs_parsed.body):
            self.process_ast(
                rhs_parsed.body[0].value,
                terms=terms,
                has_intercept=self.has_intercept
            )
        else:
            self.has_intercept[None] = '0' not in [x.strip() for x in rhs.strip().split('+')]

        self.rangf = sorted([x for x in list(self.has_intercept.keys()) if x is not None])

        self.initialize_nns()

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
                # LHS: A
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
                                for response in s.dirac_delta_responses():
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
                                for response in s.dirac_delta_responses():
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
                # Arg 2: IRF kernel definition (optional, defaults to `NN()`)

                assert not under_irf, 'IRF calls cannot be nested in the inputs to another IRF call. To compose IRFs, apply nesting in the impulse response function definition (second argument of IFR call).'
                assert 1 <= len(t.args) <= 2, 'C() takes either one or two arguments in CDR formula strings'

                arg0 = t.args[0]
                if len(t.args) == 1:  # No IRF family specified, use NN as default
                    arg1 = ast.parse('NN()').body[0].value
                else:
                    arg1 = t.args[1]

                subterms = []
                self.process_ast(
                    arg0,
                    terms=subterms,
                    has_intercept=has_intercept,
                    rangf=rangf,
                    impulses_by_name=impulses_by_name,
                    interactions_by_name=interactions_by_name,
                    under_irf=True,
                    under_interaction=under_interaction
                )
                new_subterms = []
                nn_inputs = sum(subterms, [])
                for S in subterms:
                    for s in S:
                        new = self.process_irf(
                            arg1,
                            input_irf=s,
                            ops=None,
                            rangf=rangf,
                            nn_inputs=nn_inputs,
                            impulses_by_name=impulses_by_name,
                            interactions_by_name=interactions_by_name,
                        )
                        new_subterms.append(new)
                terms.append(new_subterms)
            elif t.func.id == 'NN':
                assert len(t.args) == 1, 'NN transforms take exactly one argument in CDR formula strings'
                assert not ops, 'NN transforms cannot be dominated by ops'

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
                subterms = sum(subterms, [])
                for s in subterms:
                    assert isinstance(s, Impulse) or isinstance(s, ImpulseInteraction), 'NN transforms may only dominate nodes of type Impulse or ImpulseInteraction. Got %s.' % type(s)

                nn_config = {}
                impulses_as_inputs = True
                inputs_to_add = []
                inputs_to_drop = []
                if len(t.keywords) > 0:
                    for k in t.keywords:
                        if k.arg == 'impulses_as_inputs':
                            impulses_as_inputs = get_bool(k)
                            assert impulses_as_inputs is not None, 'Unrecognized value for impulses_as_inputs: %s' % k.value
                        elif k.arg == 'inputs_to_add':
                            assert type(k.value).__name__ == 'List', 'Non-list argument provided to keyword arg "inputs_to_add"'
                            for x in k.value.elts:
                                self.process_ast(
                                    x,
                                    terms=inputs_to_add,
                                    has_intercept=has_intercept,
                                    rangf=rangf,
                                    impulses_by_name=impulses_by_name,
                                    interactions_by_name=interactions_by_name,
                                    under_irf=True,
                                    under_interaction=under_interaction
                                )
                        elif k.arg == 'inputs_to_drop':
                            assert type(k.value).__name__ == 'List', 'Non-list argument provided to keyword arg "inputs_to_drop"'
                            for x in k.value.elts:
                                self.process_ast(
                                    x,
                                    terms=inputs_to_drop,
                                    has_intercept=has_intercept,
                                    rangf=rangf,
                                    impulses_by_name=impulses_by_name,
                                    interactions_by_name=interactions_by_name,
                                    under_irf=True,
                                    under_interaction=under_interaction
                                )
                        elif k.arg in NN_KWARG_MAP:
                            if type(k.value).__name__ == 'Constant':
                                val = k.value.value
                            elif type(k.value).__name__ == 'Str':
                                val = k.value.s
                            elif type(k.value).__name__ == 'Name':
                                val = k.value.id
                            elif type(k.value).__name__ == 'NameConstant':
                                val = k.value.value
                            elif type(k.value).__name__ == 'Num':
                                val = k.value.n
                            else:
                                raise ValueError('Unrecognized type for keyword argument value: %s.' % k.value)
                            if val is None:
                                val = 'None'
                            _nn_config = {x: val for x in NN_KWARG_MAP[k.arg]}
                            nn_config.update(_nn_config)

                inputs_to_add = sum(inputs_to_add, [])
                inputs_to_drop = sum(inputs_to_drop, [])

                new = NNImpulse(
                    subterms,
                    impulses_as_inputs=impulses_as_inputs,
                    inputs_to_add=inputs_to_add,
                    inputs_to_drop=inputs_to_drop,
                    nn_config=nn_config
                )

                if under_irf or under_interaction:
                    if new.name() in impulses_by_name:
                        new = impulses_by_name[new.name()]
                    else:
                        impulses_by_name[new.name()] = new

                    terms.append([new])
                else:
                    term_str = 'C(%s, DiracDelta())' % str(new)

                    new_ast = ast.parse(Formula.prep_formula_string(term_str)).body[0].value
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
            elif Formula.normalize_irf_family(t.func.id) in Formula.IRF_PARAMS.keys() or lcg_re.match(t.func.id) is not None:
                raise ValueError('IRF calls can only occur as inputs to C() in CDR formula strings')
            elif t.func.id == 're':
                # Regular expression
                assert len(t.args) == 1, 'Regular expression terms take exactly one string argument'
                if type(t.args[0]).__name__ == 'Str':
                    val = t.args[0].s
                else:
                    val = t.args[0].value
                assert isinstance(val, str), 'Regular expression terms take exactly one string argument'
                if under_irf or under_interaction:
                    new = Impulse(val, ops=ops, is_re=True)

                    if new.name() in impulses_by_name:
                        new = impulses_by_name[new.name()]
                    else:
                        impulses_by_name[new.name()] = new

                    terms.append([new])
                else:
                    term_name = 're("%s")' % val
                    for op in ops:
                        term_name = op + '(' + term_name + ')'

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
                # Unary transform
                assert len(t.args) <= 1, 'Only unary ops on variables supported in CDR formula strings'
                subterms = []
                func_id = t.func.id
                if func_id.startswith('pow'):
                    func_id = func_id.replace('_', '.')
                self.process_ast(
                    t.args[0],
                    terms=subterms,
                    has_intercept=has_intercept,
                    ops=[func_id] + ops,
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
            elif type(t).__name__ == 'Constant':
                t_id = str(t.value)
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
                    term_name = op + '(' + term_name + ')'

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
            input_irf,
            ops=None,
            rangf=None,
            nn_inputs=None,
            impulses_by_name=None,
            interactions_by_name=None
    ):
        """
        Process data from AST node representing part of an IRF definition and insert data into internal representation of the model.

        :param t: AST node.
        :param input_irf: ``IRFNode``, ``Impulse``, ``InterationImpulse``, or ``NNImpulse`` object; child IRF of current node
        :param ops: ``list`` of ``str``, or ``None``; ops applied to IRF. If ``None``, no ops applied
        :param rangf: ``str`` or ``None``; name of rangf for random term currently being processed, or ``None`` if currently processing fixed effects portion of model.
        :param nn_inputs: ``tuple`` or ``None``; tuple of input impulses to neural network IRF, or ``None`` if not a neural network IRF.
        :return: ``IRFNode`` object; the IRF node
        """

        if ops is None:
            ops = []
        assert t.func.id in Formula.IRF_PARAMS.keys() or Formula.is_LCG(t.func.id) is not None, 'Ill-formed model string: process_irf() called on non-IRF node'
        irf_id = None
        coef_id = None
        ranirf = True
        trainable = None
        response_params = None
        param_init={}
        nn_config = {}
        impulses_as_inputs = True
        inputs_to_add = []
        inputs_to_drop = []
        if len(t.keywords) > 0:
            for k in t.keywords:
                if k.arg == 'irf_id':
                    if type(k.value).__name__ == 'Constant':
                        assert isinstance(k.value.value, str), 'irf_id must be interpretable as a string'
                        irf_id = k.value.value
                    elif type(k.value).__name__ == 'Str':
                        irf_id = k.value.s
                    elif type(k.value).__name__ == 'Name':
                        irf_id = k.value.id
                    elif type(k.value).__name__ == 'Num':
                        irf_id = str(k.value.n)
                    else:
                        raise ValueError('Unrecognized value for irf_id: %s' % k.value)
                elif k.arg == 'coef_id':
                    if type(k.value).__name__ == 'Constant':
                        assert isinstance(k.value.value, str), 'coef_id must be interpretable as a string'
                        coef_id = k.value.value
                    elif type(k.value).__name__ == 'Str':
                        coef_id = k.value.s
                    elif type(k.value).__name__ == 'Name':
                        coef_id = k.value.id
                    elif type(k.value).__name__ == 'Num':
                        coef_id = str(k.value.n)
                    else:
                        raise ValueError('Unrecognized value for coef_id: %s' % k.value)
                elif k.arg == 'ran':
                    ranirf = get_bool(k)
                    assert ranirf is not None, 'Unrecognized value for ranirf: %s' % k.value
                elif k.arg == 'trainable':
                    assert type(k.value).__name__ == 'List', 'Non-list argument provided to keyword arg "trainable"'
                    trainable = []
                    for x in k.value.elts:
                        if type(x).__name__ == 'Constant':
                            assert isinstance(x.value, str), 'trainable variable must be interpretable as a string'
                            trainable.append(x.value)
                        elif type(x).__name__ == 'Name':
                            trainable.append(x.id)
                        elif type(x).__name__ == 'Str':
                            trainable.append(x.s)
                        else:
                            raise ValueError('Unrecognized value for element of trainable: %s' % x)
                elif k.arg == 'response_params':
                    assert type(k.value).__name__ == 'List', 'Non-list argument provided to keyword arg "response_params"'
                    response_params = []
                    for x in k.value.elts:
                        if type(x).__name__ == 'Constant':
                            assert isinstance(x.value, str), 'response_params item must be interpretable as a string'
                            x = x.value
                        elif type(x).__name__ == 'Name':
                            x = x.id
                        elif type(x).__name__ == 'Str':
                            x = x.s
                        else:
                            raise ValueError('Unrecognized value for element of response_params: %s' % x)
                        x = x.split('_')
                        if len(x) == 1:
                            x = (None, x[0])
                        x = tuple(x)
                        if len(x) != 2:
                            raise ValueError('Element of response_params must either be the name of a distributional parameter or a "_"-delimited pair of distribution name and parameter name.')
                        assert x[0] is None or x[0] in Formula.RESPONSE_DISTRIBUTIONS, 'Distribution name %s not currently supported' % x[0]
                        if x[0] is not None:
                            assert x[1] in Formula.RESPONSE_DISTRIBUTIONS[x[0]]['params'], 'Parameter %s not found for distribution %s.' % (x[1], x[0])

                        response_params.append(x)

                elif t.func.id == 'NN' and k.arg == 'impulses_as_inputs':
                    impulses_as_inputs = get_bool(k)
                    assert impulses_as_inputs is not None, 'Unrecognized value for impulses_as_inputs: %s' % k.value
                elif t.func.id == 'NN' and k.arg == 'inputs_to_add':
                    assert type(k.value).__name__ == 'List', 'Non-list argument provided to keyword arg "inputs_to_add"'
                    for x in k.value.elts:
                        self.process_ast(
                            x,
                            terms=inputs_to_add,
                            rangf=rangf,
                            impulses_by_name=impulses_by_name,
                            interactions_by_name=interactions_by_name,
                            under_irf=True
                        )
                elif t.func.id == 'NN' and k.arg == 'inputs_to_drop':
                    assert type(k.value).__name__ == 'List', 'Non-list argument provided to keyword arg "inputs_to_drop"'
                    for x in k.value.elts:
                        self.process_ast(
                            x,
                            terms=inputs_to_drop,
                            rangf=rangf,
                            impulses_by_name=impulses_by_name,
                            interactions_by_name=interactions_by_name,
                            under_irf=True
                        )
                elif t.func.id == 'NN' and k.arg in NN_KWARG_MAP:
                    if type(k.value).__name__ == 'Constant':
                        val = k.value.value
                    elif type(k.value).__name__ == 'Str':
                        val = k.value.s
                    elif type(k.value).__name__ == 'Name':
                        val = k.value.id
                    elif type(k.value).__name__ == 'NameConstant':
                        val = k.value.value
                    elif type(k.value).__name__ == 'Num':
                        val = k.value.n
                    else:
                        raise ValueError('Unrecognized type for keyword argument value: %s.' % k.value)
                    if val is None:
                        val = 'None'
                    _nn_config = {x: val for x in NN_KWARG_MAP[k.arg]}
                    nn_config.update(_nn_config)
                else:
                    if type(k.value).__name__ == 'Constant':
                        param_init[k.arg] = k.value.value
                    elif type(k.value).__name__ == 'Num':
                        param_init[k.arg] = k.value.n
                    elif type(k.value).__name__ == 'UnaryOp':
                        assert type(k.value.op).__name__ == 'USub', 'Invalid operator provided to to IRF parameter "%s"' %k.arg
                        if type(k.value.operand).__name__ == 'Constant':
                            param_init[k.arg] = -k.value.operand.value
                        elif type(k.value.operand).__name__ == 'Num':
                            param_init[k.arg] = -k.value.operand.n
                        else:
                            raise ValueError('Non-numeric initialization provided to IRF parameter "%s"' % k.arg)
                    else:
                        raise ValueError('Non-numeric initialization provided to IRF parameter "%s"' %k.arg)

        inputs_to_add = sum(inputs_to_add, [])
        inputs_to_drop = sum(inputs_to_drop, [])

        if isinstance(input_irf, IRFNode):
            new = IRFNode(
                family=t.func.id,
                irfID=irf_id,
                ops=ops,
                fixed=rangf is None,
                rangf=rangf if ranirf else None,
                nn_impulses=nn_inputs,
                param_init=param_init,
                nn_config=nn_config,
                impulses_as_inputs=impulses_as_inputs,
                inputs_to_add=inputs_to_add,
                inputs_to_drop=inputs_to_drop,
                trainable=trainable,
                response_params_list=response_params
            )

            new.add_child(input_irf)

            if len(t.args) > 0:
                assert len(t.args) == 1, 'Ill-formed model string: IRF can take at most 1 positional argument'
                p = self.process_irf(
                    t.args[0],
                    input_irf= new,
                    nn_inputs=nn_inputs,
                    impulses_by_name=impulses_by_name,
                    interactions_by_name=interactions_by_name,
                )
            else:
                p = self.t

            p.add_child(new)

        else:
            new = IRFNode(
                family='Terminal',
                impulse=input_irf,
                coefID=coef_id,
                fixed=rangf is None,
                rangf=rangf,
                param_init=param_init,
                trainable=trainable
            )

            p = self.process_irf(
                t,
                input_irf=new,
                rangf=rangf,
                nn_inputs=nn_inputs,
                impulses_by_name=impulses_by_name,
                interactions_by_name=interactions_by_name,
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

        if op in ['c', 'c.']:
            out = c(arr)
        elif op in ['z', 'z.']:
            out = z(arr)
        elif op in ['s', 's.']:
            out = s(arr)
        elif op == 'log':
            out = np.log(np.maximum(arr, 1e-12))
        elif op == 'log1p':
            out = np.log(arr + 1)
        elif op == 'exp':
            out = np.exp(arr)
        elif op.startswith('add'):
            x = float(op[3:])
            out = arr + x
        elif op.startswith('subtract'):
            x = float(op[8:])
            out = arr - x
        elif op.startswith('multiply'):
            x = float(op[8:])
            out = arr * x
        elif op.startswith('divide'):
            x = float(op[6:])
            out = arr / x
        elif op.startswith('pow'):
            exponent = float(op[3:])
            out = arr ** exponent
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
                if type(impulse).__name__ in ('ImpulseInteraction', 'NNImpulse'):
                    _X, expanded_impulses, expanded_atomic_impulses = impulse.expand_categorical(_X)
                    for x in expanded_atomic_impulses:
                        for a in x:
                            _X = self.apply_ops(a, _X)
                    for x in expanded_impulses:
                        if x.name() not in _X:
                            _X[x.id] = _X[[y.name() for y in x.atomic_impulses]].product(axis=1)
            else:
                if type(impulse).__name__ == ('ImpulseInteraction', 'NNImpulse'):
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
                            if x.id not in _Y:
                                print(str(self))
                            assert x.id in _Y, 'Impulse %s not found in data. Either it is missing from all of the predictor files X, or (if response aligned) it is missing from at least one of the response files Y.' % x.name()
                            Y[i] = self.apply_ops(x, _Y)
                        if X_in_Y_names is None:
                            X_in_Y_names = []
                        if x.name() not in X_in_Y_names:
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
                    if impulse.name() not in X_in_Y_names:
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
                if (term['irf'], term['nn_key']) in new_terms:
                    new_terms[(term['irf'], term['nn_key'])]['impulses'] += term['impulses']
                else:
                    new_terms[(term['irf'], term['nn_key'])] = term
            new_terms = [new_terms[x] for x in sorted(list(new_terms.keys()), key=lambda x: x[0])]
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
                if (term['irf'], term['nn_key']) in new_terms:
                    new_terms[(term['irf'], term['nn_key'])]['impulses'] += term['impulses']
                else:
                    new_terms[(term['irf'], term['nn_key'])] = term
            new_terms = [new_terms[x] for x in sorted(list(new_terms.keys()), key=lambda x: x[0])]
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

    def re_transform(self, X):
        """
        Get transformed formula with regex predictors expanded based on matches to the columns in **X**.

        :param X: list of ``pandas`` tables; input data.
        :return: ``Formula``; transformed ``Formula`` object
        """

        new_t = self.t.re_transform(X)[0]
        new_formstring = self.to_string(t=new_t)
        new_form = Formula(new_formstring)
        return new_form

    def initialize_nns(self):
        """
        Initialize a dictionary mapping ids to metadata for all NN components in this CDR model

        :return: ``dict``; mapping from NN ``str`` id to ``NN`` object storing metadata for that NN.
        """
        nn_meta_by_key = self.t.nns_by_key()
        nns_by_key = {}

        for key in nn_meta_by_key:
            nodes = []
            nn_types = []
            rangf = []
            nn_config = {}
            for node, gf in nn_meta_by_key[key]:
                nodes.append(node)
                nn_config.update(node.nn_config)
                if isinstance(node, IRFNode):
                    nn_types.append('irf')
                elif isinstance(node, NNImpulse):
                    nn_types.append('impulse')
                else:
                    raise ValueError('Got unsupported object of type "%s" associated with NN key "%s".' % (type(node), key))
                rangf += gf
            assert nodes, 'NN key "%s" must be associated with at least one node in the tree.' % key
            nodes = set(nodes)
            nn_types = set(nn_types)
            rangf = set(rangf)
            assert len(nn_types) == 1, 'Each NN key may only have 1 type. Got > 1 types for NN key "%s".' % key
            nn_inputs = sorted(nodes, key=lambda x: x.name())
            nn_type = tuple(nn_types)[0]
            rangf = list(rangf)

            nn = NN(nn_inputs, nn_type, rangf=rangf, nn_key=key, nn_config=nn_config)
            nns_by_key[key] = nn

        keys = sorted(nns_by_key)
        nn_impulse_keys = [key for key in keys if nns_by_key[key].nn_type == 'impulse']
        nn_irf_keys = [key for key in keys if nns_by_key[key].nn_type == 'irf']
        ids = ['NN%d' % (i + 1) for i in range(len(nn_impulse_keys))] + ['NNirf%d' % (i + 1) for i in range(len(nn_irf_keys))]

        self.nns_by_id = {k: nns_by_key[keys[i]] for i, k in enumerate(ids)}

    def __str__(self):
        return self.to_string()

class Impulse(object):
    """
    Data structure representing an impulse in a CDR model.

    :param name: ``str``; name of impulse
    :param ops: ``list`` of ``str``, or ``None``; ops to apply to impulse. If ``None``, no ops.
    :param is_re: ``bool``; whether impulse is a regular expression search pattern
    """

    def __init__(self, name, ops=None, is_re=False):
        if ops is None:
            ops = []
        self.ops = ops[:]
        self.is_re = is_re
        if self.is_re:
            self.name_str = 're("%s")' % name
        else:
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
                impulses = [Impulse('_'.join([self.id, pythonize_string(str(val))]), ops=self.ops, is_re=self.is_re) for val in vals]
                expanded_value_names = [str(val) for val in vals]
                for j in range(len(impulses)):
                    x = impulses[j]
                    val = expanded_value_names[j]
                    if x.id not in _X:
                        _X[x.id] = (_X[self.id] == val).astype('float')
                X[i] = _X
                break

        if delistify:
            X = X[0]

        return X, impulses

    def get_matcher(self):
        """
        Return a compiled regex matcher to compare to data columns

        :return: ``re`` object
        """

        if self.id.endswith('$'):
            pattern = self.id
        else:
            pattern = self.id + '$'
        matcher = re.compile(pattern)
        return matcher

    def expand_re(self, X):
        """
        Expand any regular expression predictors in **X** into a sequence of all matching columns.

        :param X: list of ``pandas`` tables; input data
        :return: ``list`` of ``Impulse``; list of expanded ``Impulse`` objects
        """

        if not isinstance(X, list):
            X = [X]

        impulses = []

        if self.id.endswith('$'):
            pattern = self.id
        else:
            pattern = self.id + '$'
        matcher = re.compile(pattern)

        for i in range(len(X)):
            _X = X[i]
            for col in _X:
                if matcher.match(col):
                    impulses.append(
                        Impulse(
                            col,
                            ops=self.ops,
                            is_re=False
                        )
                    )

        return impulses

    def is_nn_impulse(self):
        """
        Type check for whether impulse represents an NN transformation of impulses.

        :return: ``False``
        """

        return False


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

        return self.atomic_impulses[:]

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

    def expand_re(self, X):
        """
        Expand any regular expression predictors in **X** into a sequence of all matching columns.

        :param X: list of ``pandas`` tables; input data
        :return: 2-tuple of ``list`` of ``ImpulseInteraction``, ``list`` of ``list`` of ``Impulse``; list of expanded ``ImpulseInteraction`` objects, list of lists of expanded ``Impulse`` objects, one list for each interaction.
        """

        if not isinstance(X, list):
            X = [X]

        expanded_atomic_impulses = []
        for x in self.impulses():
            X, expanded_atomic_impulses_cur = x.expand_re(X)
            expanded_atomic_impulses.append(expanded_atomic_impulses_cur)
        expanded_interaction_impulses = [ImpulseInteraction(x, ops=self.ops) for x in itertools.product(*expanded_atomic_impulses)]

        return expanded_interaction_impulses, expanded_atomic_impulses

    def is_nn_impulse(self):
        """
        Type check for whether impulse represents an NN transformation of impulses.

        :return: ``False``
        """

        return False


class NNImpulse(object):
    """
    Data structure representing a feedforward neural network transform of one or more impulses in a CDR model.

    :param impulses: ``list`` of ``Impulse``; impulses to transform.
    :param impulses_as_inputs: ``bool``; whether to include impulses as NN inputs.
    :param inputs_to_add: ``list`` of ``Impulse`` or ``None``; extra impulses to add to NN input.
    :param inputs_to_drop: ``list`` of ``Impulse`` or ``None``;  output impulses to drop from NN input.
    :param nn_config: ``dict`` or ``None``; map of NN config fields to their values for this NN node.
    """

    def __init__(self, impulses, impulses_as_inputs=True, inputs_to_add=None, inputs_to_drop=None, nn_config=None):
        self.atomic_impulses = []
        names = set()
        for x in impulses:
            names.add(x.name())
            self.atomic_impulses.append(x)
        self.nn_impulses = self.atomic_impulses[:]
        self.impulses_as_inputs = impulses_as_inputs

        if self.impulses_as_inputs:
            nn_inputs = self.atomic_impulses[:]
        else:
            nn_inputs = []
        added = []
        if inputs_to_add is None:
            inputs_to_add = []
        for x in inputs_to_add:
            found = False
            for y in nn_inputs:
                if x.name() == y.name():
                    found = True
                    break
            if found:
                stderr('WARNING: Input to be added "%s" was already present. Skipping...\n' % x.name())
            else:
                added.append(x)
                nn_inputs.append(x)
        self.inputs_added = added

        if inputs_to_drop is None:
            inputs_to_drop = []
        dropped = []
        for x in inputs_to_drop:
            success = False
            for i, y in enumerate(nn_inputs):
                success = False
                if x.name() == y.name():
                    success = True
                    break
            if success:
                dropped.append(nn_inputs.pop(i))
            else:
                stderr('WARNING: Input to be dropped "%s" was not found. Skipping...\n' % x.name())
        self.inputs_dropped = dropped

        if nn_config is None:
            nn_config = {}
        self.nn_config = nn_config

        self.nn_inputs = tuple(sorted([x for x in nn_inputs], key=lambda x: x.name()))
        nn_args = [' + '.join([str(x) for x in self.impulses()])]
        if not self.impulses_as_inputs:
            nn_args.append('impulses_as_inputs=False')
        if self.inputs_added:
            key = 'inputs_to_add'
            val = []
            for x in self.inputs_added:
                val.append(x.name())
            val = '[%s]' % ', '.join([x for x in val])
            nn_args.append('%s=%s' % (key, val))
        if self.inputs_dropped:
            key = 'inputs_to_drop'
            val = []
            for x in self.inputs_dropped:
                val.append(x.name())
            val = '[%s]' % ', '.join([x for x in val])
            nn_args.append('%s=%s' % (key, val))
        for key in self.nn_config:
            val = self.nn_config[key]
            if isinstance(val, str) and val != 'None':
                val = '"%s"' % val
            nn_args.append('%s=%s' % (key, val))

        self.name_str = 'NN(%s)' % ', '.join(nn_args)
        self.ops = []

        self.id = ':'.join([x.id for x in sorted(self.atomic_impulses, key=lambda x: x.id)])

        self.response_params = None
    
    def __setstate__(self, state):
        self.response_params = state.pop('response_params', None)
        for key in state:
            if key != 'nn_key':
                setattr(self, key, state[key])

    @property
    def nn_key(self):
        out = 'impulseNN_' + '_'.join([x.name() for x in self.nn_inputs])
        if self.inputs_dropped:
            out += '_dropped:' + '_'.join([x.name() for x in self.inputs_dropped])
        if self.inputs_added:
            out += '_added:' + '_'.join([x.name() for x in self.inputs_added])
        return out

    def __str__(self):
        return self.name_str

    def name(self):
        """
        Get name of NN impulse.

        :return: ``str``; name.
        """

        return self.name_str

    def impulses(self):
        """
        Get list of output impulses dominated by NN.

        :return: ``list`` of ``Impulse``; impulses dominated by NN.
        """

        return self.atomic_impulses[:]

    def expand_categorical(self, X):
        """
        Expand any categorical predictors in **X** into 1-hot columns.

        :param X: list of ``pandas`` tables; input data.
        :return: 3-tuple of ``pandas`` table, ``list`` of ``NNImpulse``, ``list`` of ``list`` of ``Impulse``; expanded data, list of expanded ``NNImpulse`` objects, list of lists of expanded ``Impulse`` objects, one list for each interaction.
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
        expanded_interaction_impulses = [
            NNImpulse(
                sum([expanded_atomic_impulses], []),
                impulses_as_inputs=self.impulses_as_inputs,
                inputs_to_add=self.inputs_added,
                inputs_to_drop=self.inputs_dropped,
                nn_config=self.nn_config
            )
        ]

        if delistify:
            X = X[0]

        return X, expanded_interaction_impulses, expanded_atomic_impulses

    def expand_re(self, X):
        """
        Expand any regular expression predictors in **X** into a sequence of all matching columns.

        :param X: list of ``pandas`` tables; input data
        :return: 2-tuple of ``list`` of ``ImpulseInteraction``, ``list`` of ``list`` of ``Impulse``; list of expanded ``ImpulseInteraction`` objects, list of lists of expanded ``Impulse`` objects, one list for each interaction.
        """

        if not isinstance(X, list):
            X = [X]

        expanded_atomic_impulses = []
        for x in self.impulses():
            X, expanded_atomic_impulses_cur = x.expand_re(X)
            expanded_atomic_impulses.append(expanded_atomic_impulses_cur)
        expanded_interaction_impulses = [
            NNImpulse(
                sum([expanded_atomic_impulses], []),
                impulses_as_inputs=self.impulses_as_inputs,
                inputs_to_add=self.inputs_added,
                inputs_to_drop=self.inputs_dropped,
                nn_config=self.nn_config
            )
        ]

        return expanded_interaction_impulses, expanded_atomic_impulses

    def is_nn_impulse(self):
        """
        Type check for whether impulse represents an NN transformation of impulses.

        :return: ``True``
        """

        return True


class NN(object):
    """
    Data structure representing a neural network within a CDR model.

    :param nodes: ``list`` of ``IRFNode``, and/or ``NNImpulse`` objects; nodes associated with this NN
    :param nn_type: ``str``; name of NN type (``'irf'`` or ``'impulse'``).
    :param rangf: ``str`` or list of ``str``; random grouping factors for which to build random effects for this NN.
    :param nn_type: ``str`` or ``None``; key uniquely identifying this NN node (constructed automatically if ``None``).
    :param nn_config: ``dict`` or ``None``; map of NN config fields to their values for this NN node.
    """

    def __init__(self, nodes, nn_type, rangf=None, nn_key=None, nn_config=None):
        assert nn_type in ('irf', 'impulse'), 'nn_type must be either "irf" or "impulse". Got %s.' % nn_type
        _nodes = []
        nn_impulses = set()
        nn_inputs = set()
        inputs_added = set()
        inputs_dropped = set()
        names = set()
        response_params = None
        for x in nodes:
            names.add(x.name())
            _nodes.append(x)
            nn_impulses |= set(x.nn_impulses)
            nn_inputs |= set(x.nn_inputs)
            inputs_added |= set(x.inputs_added)
            inputs_dropped |= set(x.inputs_dropped)
            if x.response_params:
                if response_params is None:
                    response_params = {}
                for k in x.response_params:
                    if k not in response_params:
                        response_params[k] = set()
                    response_params[k] |= x.response_params[k]
        self.nn_impulses = tuple(sorted(list(nn_impulses), key=lambda x: x.name()))
        self.nn_inputs = tuple(sorted(list(nn_inputs), key=lambda x: x.name()))
        self.inputs_added = tuple(sorted(list(inputs_added), key=lambda x: x.name()))
        self.inputs_dropped = tuple(sorted(list(inputs_dropped), key=lambda x: x.name()))
        self.response_params = response_params
        self.nodes = tuple(sorted([x for x in _nodes], key=lambda x: x.name()))
        self.nn_type = nn_type
        self.name_str = ', '.join([str(x) for x in self.nodes])
        if nn_key is None:
            self.nn_key = '%sNN_' % nn_type + '_'.join([x.name() for x in self.nodes])
        else:
            self.nn_key = nn_key
        if nn_config is None:
            nn_config = {}
        self.nn_config = nn_config
        if not isinstance(rangf, list):
            rangf = [rangf]
        self.rangf = rangf

    def __setstate__(self, state):
        self.response_params = state.pop('response_params', None)
        for key in state:
            setattr(self, key, state[key])

    def all_impulse_names(self):
        """
        Get list of all impulse names associated with this NN component.

        :return: ``list`` of ``str``: All impulse names associated with this NN component.
        """

        return [x.name() for x in self.nn_impulses + self.inputs_added]

    def input_impulse_names(self):
        """
        Get list of input impulse names associated with this NN component.

        :return: ``list`` of ``str``: Input impulse names associated with this NN component.
        """

        return [x.name() for x in self.nn_inputs]

    def output_impulse_names(self):
        """
        Get list of output impulse names associated with this NN component (NN IRF only).

        :return: ``list`` of ``str``: Output impulse names associated with this NN component.
        """

        if self.nn_type == 'irf':
            return [x.name() for x in self.nn_impulses]
        return []

    def __str__(self):
        return 'NN; nn_key: %s; nn_type: %s; nodes: %s' % (self.nn_key, self.nn_type, ', '.join([x.name() for x in self.nodes]))

    def name(self):
        """
        Get name of NN.

        :return: ``str``; name.
        """

        return self.name_str


class ResponseInteraction(object):
    """
    Data structure representing an interaction of response-aligned variables (containing at least one IRF-convolved impulse) in a CDR model.

    :param responses: ``list`` of terminal ``IRFNode``, ``Impulse``, and/or ``ImpulseInteraction`` objects; responses to interact.
    :param rangf: ``str`` or list of ``str``; random grouping factors for which to build random effects for this interaction.
    """

    def __init__(self, responses, rangf=None):
        self.atomic_responses = []
        for x in responses:
            assert (type(x).__name__ == 'IRFNode' and x.terminal()) or type(x).__name__ in ['Impulse', 'ImpulseInteraction', 'ResponseInteraction', 'NNImpulse'], 'All inputs to ResponseInteraction must be either terminal IRFNode, Impulse, ImpulseInteraction, ResponseInteraction, or NNImpulse objects. Got %s.' % type(x).__name__
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

    def nn_impulse_responses(self):
        """
        Get list of NN impulse terms dominated by interaction.

        :return: ``list`` of ``NNImpulse`` objects; NN impulse terms dominated by interaction.
        """

        return [x for x in self.atomic_responses if type(x).__name__ == 'NNImpulse']

    def dirac_delta_responses(self):
        """
        Get list of response-aligned Dirac delta variables dominated by interaction.

        :return: ``list`` of ``Impulse`` and/or ``ImpulseInteraction`` objects; Dirac delta variables dominated by interaction.
        """

        return [x for x in self.atomic_responses if type(x).__name__ in ('Impulse', 'ImpulseInteraction')]

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
    :param fixed: ``bool``; Whether node exists in the model's fixed effects structure.
    :param rangf: ``list`` of ``str``, ``str``, or ``None``; names of any random grouping factors associated with the node.
    :param nn_impulses: ``tuple`` or ``None``; tuple of input impulses to neural network IRF, or ``None`` if not a neural network IRF.
    :param nn_config: ``dict`` or ``None``; dictionary of settings for NN IRF component.
    :param impulses_as_inputs: ``bool``; whether to include impulses in input of a neural network IRF.
    :param inputs_to_add: ``list`` of ``Impulse``/``NNImpulse`` or ``None``; list of impulses to add to input of neural network IRF.
    :param inputs_to_drop: ``list`` of ``Impulse``/``NNImpulse`` or ``None``; list of impulses to remove from input of neural network IRF (keeping them in output).
    :param param_init: ``dict``; map from parameter names to initial values, which will also be used as prior means.
    :param trainable: ``list`` of ``str``, or ``None``; trainable parameters at this node. If ``None``, all parameters are trainable.
    :param response_params_list: ``list`` of 2-``tuple`` of ``str``, or ``None``; Response distribution parameters modeled by this IRF, with each parameter represented as a pair (DIST_NAME, PARAM_NAME). DIST_NAME can be ``None``, in which case the IRF will apply to any distribution parameter matching PARAM_NAME.
    """

    def __init__(
            self,
            family=None,
            impulse=None,
            p=None,
            irfID=None,
            coefID=None,
            ops=None,
            fixed=True,
            rangf=None,
            nn_impulses=None,
            nn_config=None,
            impulses_as_inputs=True,
            inputs_to_add=None,
            inputs_to_drop=None,
            param_init=None,
            trainable=None,
            response_params_list=None
    ):
        family = Formula.normalize_irf_family(family)
        if family is None or family in ['Terminal', 'DiracDelta']:
            assert irfID is None, 'Attempted to tie parameters (irf_id=%s) on parameter-free IRF node (family=%s)' % (irfID, family)
        if family != 'Terminal':
            assert coefID is None, 'Attempted to set coef_id=%s on non-terminal IRF node (family=%s)' % (coefID, family)
            assert impulse is None, 'Attempted to attach impulse (%s) to non-terminal IRF node (family=%s)' % (impulse, family)
        if family is None:
            self.ops = []
            self.impulse = None
            self.family = None
            self.irfID = None
            self.coefID = None
            self.fixed = fixed
            self.rangf = []
            self.impulses_as_inputs = True
            self.nn_impulses = tuple()
            self.inputs_added = []
            self.inputs_dropped = []
            self.nn_config = {}
            self.param_init = {}
        else:
            self.ops = [] if ops is None else ops[:]
            self.impulse = impulse
            self.family = family
            self.irfID = irfID
            self.coefID = coefID
            self.fixed = fixed
            self.rangf = [] if rangf is None else sorted(rangf) if isinstance(rangf, list) else [rangf]
            self.impulses_as_inputs = impulses_as_inputs
            if family == 'NN':
                assert nn_impulses, 'Parameter nn_impulses must be provided to neural network IRFs'
                self.nn_impulses = tuple(sorted([x for x in nn_impulses], key=lambda x: x.name()))

                if self.impulses_as_inputs:
                    nn_inputs = list(self.nn_impulses)
                else:
                    nn_inputs = []
                added = []
                if inputs_to_add is None:
                    inputs_to_add = []
                for x in inputs_to_add:
                    found = False
                    for y in nn_inputs:
                        if x.name() == y.name():
                            found = True
                            break
                    if found:
                        stderr('WARNING: Input to be added "%s" was already present. Skipping...\n' % x.name())
                    else:
                        added.append(x)
                        nn_inputs.append(x)
                self.inputs_added = added

                if inputs_to_drop is None:
                    inputs_to_drop = []
                dropped = []
                for x in inputs_to_drop:
                    success = False
                    for i, y in enumerate(nn_inputs):
                        success = False
                        if x.name() == y.name():
                            success = True
                            break
                    if success:
                        dropped.append(nn_inputs.pop(i))
                    else:
                        stderr('WARNING: Input to be dropped "%s" was not found. Skipping...\n' % x.name())
                self.inputs_dropped = dropped

                self.nn_inputs = tuple(sorted([x for x in nn_inputs], key=lambda x: x.name()))

                if nn_config is None:
                    nn_config = {}
                self.nn_config = nn_config
            else:
                self.nn_impulses = tuple()
                self.inputs_added = []
                self.inputs_dropped = []
                self.nn_config = {}

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
        self.response_params_list = response_params_list
        _response_params = None
        if response_params_list is not None:
            assert family == 'NN', 'response_params is currently only supported for IRFs of type ``NN``'
            _response_params = {}
            for dist_name, param_name in response_params_list:
                if dist_name not in _response_params:
                    _response_params[dist_name] = set()
                _response_params[dist_name].add(param_name)
        self.response_params = _response_params

        self.children = []
        self.p = p
        if self.p is not None:
            self.p.add_child(self)

        self.interaction_list = []

    def __setstate__(self, state):
        self.response_params = state.pop('response_params', None)
        for key in state:
            if key != 'nn_key':
                setattr(self, key, state[key])

    @property
    def nn_key(self):
        out = None
        if self.family == 'NN':
            if self.irfID:
                out = self.irfID
            else:
                out = 'irfNN_' + '_'.join([x.name() for x in self.nn_impulses])
                if self.inputs_dropped:
                    out += '_dropped:' + '_'.join([x.name() for x in self.inputs_dropped])
                if self.inputs_added:
                    out += '_added:' + '_'.join([x.name() for x in self.inputs_added])
                if self.response_params:
                    vals = []
                    for key in self.response_params:
                        if key is None:
                            val = 'any:' + '-'.join(sorted(self.response_params[key]))
                        else:
                            val = key + ':' + '-'.join([x for x in Formula.RESPONSE_DISTRIBUTIONS[key]['params'] \
                                                        if x in self.response_params[key]])
                        vals.append(val)
                    out += '_' + '_'.join(vals)
        return out

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
            if self.family == 'NN':
                assert t.terminal(), 'Neural network IRFs cannot dominate other IRF types in the network tree'
            if t.family == 'NN':
                assert self.family is None, 'Neural network IRFs cannot be dominated by other IRF types in the network tree'
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
            out = '.'.join([self.family] + self.impulse_names(include_nn_inputs=False))
            if self.inputs_dropped:
                out += '_dropped:' + '_'.join([x.name() for x in self.inputs_dropped])
            if self.inputs_added:
                out += '_added:' + '_'.join([x.name() for x in self.inputs_added])
            if self.response_params:
                vals = []
                for key in self.response_params:
                    if key is None:
                        val = 'any:' + '-'.join(sorted(self.response_params[key]))
                    else:
                        val = key + ':' + '-'.join([x for x in Formula.RESPONSE_DISTRIBUTIONS[key]['params'] \
                                                    if x in self.response_params[key]])
                    vals.append(val)
                out += '_' + '_'.join(vals)
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
            if len(self.param_init) > 0:
                params.append(', '.join(['%s=%s' % (x, self.param_init[x]) for x in self.param_init]))
            if set(self.trainable) != set(Formula.irf_params(self.family)):
                params.append('trainable=%s' % self.trainable)
            if self.response_params:
                vals = []
                for dist_name in self.response_params:
                    if dist_name is None:
                        param_names = sorted(self.response_params[dist_name])
                    else:
                        param_names = [x for x in Formula.RESPONSE_DISTRIBUTIONS[dist_name]['params'] \
                                       if x in self.response_params[dist_name]]
                    for param_name in param_names:
                        if dist_name is None:
                            vals.append(param_name)
                        else:
                            vals.append('_'.join((dist_name, param_name)))
                params.append('response_params=[%s]' % ','.join(vals))
            if self.family == 'NN':
                if not self.impulses_as_inputs:
                    params.append('impulses_as_inputs=False')
                if self.inputs_added:
                    key = 'inputs_to_add'
                    val = []
                    for x in self.inputs_added:
                        val.append(x.name())
                    val = '[%s]' % ', '.join([x for x in val])
                    params.append('%s=%s' % (key, val))
                if self.inputs_dropped:
                    key = 'inputs_to_drop'
                    val = []
                    for x in self.inputs_dropped:
                        val.append(x.name())
                    val = '[%s]' % ', '.join([x for x in val])
                    params.append('%s=%s' % (key, val))
                for key in self.nn_config:
                    val = self.nn_config[key]
                    if isinstance(val, str) and val != 'None':
                        val = '"%s"' % val
                    params.append('%s=%s' % (key, val))

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

    def nns_by_key(self, nns_by_key=None):
        """
        Get a dict mapping NN keys to objects associated with them.

        :param keys: ``dict`` or ``None``; dictionary to modify. Empty if ``None``.

        :return: ``dict``; map from string keys to ``list`` of associated ``IRFNode`` and/or ``NNImpulse`` objects.
        """

        if nns_by_key is None:
            nns_by_key = {}

        if self.family == 'NN':
            if self.nn_key not in nns_by_key:
                nns_by_key[self.nn_key] = [(self, self.rangf[:])]
            else:
                nns_by_key[self.nn_key].append((self, self.rangf[:]))

        if self.terminal():
            for x in self.impulses(include_interactions=True, include_nn=True):
                if isinstance(x, NNImpulse):
                    if x.nn_key not in nns_by_key:
                        nns_by_key[x.nn_key] = [(x, self.rangf[:])]
                    else:
                        nns_by_key[x.nn_key].append((x, self.rangf[:]))

        for c in self.children:
            nns_by_key = c.nns_by_key(nns_by_key=nns_by_key)

        return nns_by_key

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

    def impulses(self, include_interactions=False, include_nn=False, include_nn_inputs=True):
        """
        Get alphabetically sorted list of impulses dominated by node.
    
        :param include_interactions: ``bool``; whether to return impulses defined by interaction terms.
        :param include_nn: ``bool``; whether to return NN transformations of impulses.
        :param include_nn_inputs: ``bool``; whether to return input impulses to NN transformations.

        :return: ``list`` of ``Impulse``; impulses dominated by node.
        """

        impulses_by_name = self.impulses_by_name(
            include_interactions=include_interactions,
            include_nn=include_nn,
            include_nn_inputs=include_nn_inputs
        )

        out = [impulses_by_name[x] for x in sorted(impulses_by_name.keys())]

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

    def impulse_names(self, include_interactions=False, include_nn=False, include_nn_inputs=True):
        """
        Get list of names of impulses dominated by node.

        :param include_interactions: ``bool``; whether to return impulses defined by interaction terms.
        :param include_nn: ``bool``; whether to return NN transformations of impulses.
        :param include_nn_inputs: ``bool``; whether to return input impulses to NN transformations.
       
        :return: ``list`` of ``str``; names of impulses dominated by node.
        """

        return sorted(self.impulses_by_name(
            include_interactions=include_interactions,
            include_nn=include_nn,
            include_nn_inputs=include_nn_inputs
        ).keys())

    def impulses_by_name(self, include_interactions=False, include_nn=False, include_nn_inputs=True):
        """
        Get dictionary mapping names of impulses dominated by node to their corresponding impulses.

        :param include_interactions: ``bool``; whether to return impulses defined by interaction terms.
        :param include_nn: ``bool``; whether to return NN transformations of impulses.
        :param include_nn_inputs: ``bool``; whether to return input impulses to NN transformations.

        :return: ``list`` of ``Impulse``; impulses dominated by node.
        """

        impulse_set = self.impulse_set(
            include_interactions=include_interactions,
            include_nn=include_nn,
            include_nn_inputs=include_nn_inputs
        )

        out = {x.name(): x for x in impulse_set}

        return out

    def impulse_set(self, include_interactions=False, include_nn=False, include_nn_inputs=True, out=None):
        """
        Get set of impulses dominated by node.

        :param include_interactions: ``bool``; whether to return impulses defined by interaction terms.
        :param include_nn: ``bool``; whether to return NN transformations of impulses.
        :param include_nn_inputs: ``bool``; whether to return input impulses to NN transformations.
        :param ``set`` or ``None``; initial dictionary to modify.

        :return: ``list`` of ``Impulse``; impulses dominated by node.
        """

        if out is None:
            out = set()

        if self.terminal():
            if include_nn or not self.impulse.is_nn_impulse():
                out.add(self.impulse)
            if include_nn_inputs:
                if isinstance(self.impulse, NNImpulse):
                    for impulse in self.impulse.nn_inputs:
                        out.add(impulse)
            if include_interactions:
                for interaction in self.interactions():
                    for response in interaction.responses():
                        if isinstance(response, IRFNode):
                            out.add(response.impulse)
                        elif isinstance(response, ImpulseInteraction) or (include_nn_inputs and isinstance(response, NNImpulse)):
                            for subresponse in response.impulses():
                                out.add(subresponse)
                        elif isinstance(response, Impulse):
                            out.add(response)
                        else:
                            raise ValueError('Unsupported type "%s" for input to interaction' % type(response).__name__)
        else:
            for c in self.children:
                c.impulse_set(
                    include_interactions=include_interactions,
                    include_nn=include_nn,
                    include_nn_inputs=include_nn_inputs,
                    out=out
                )
                if self.family == 'NN':
                    if include_nn_inputs:
                        for impulse in self.inputs_added:
                            if include_nn or not impulse.is_nn_impulse():
                                out.add(impulse)

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
                                        vals = sorted(_X[response.id].unique())[1:]
                                        expansion = [Impulse('_'.join([response.id, pythonize_string(str(val))]), ops=response.ops, is_re=response.is_re) for val in vals]
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
                                            vals = sorted(_X[subresponse.id].unique())[1:]
                                            expansion = [
                                                Impulse('_'.join([subresponse.id, pythonize_string(str(val))]), ops=subresponse.ops, is_re=subresponse.is_re)
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
                                    vals = sorted(_X[x.id].unique())[1:]
                                    expansion = [Impulse('_'.join([x.id, pythonize_string(str(val))]), ops=x.ops, is_re=x.is_re) for val in vals]
                                    break
                            assert found, 'Impulse %s not found in data.' % x.id
                        else:
                            expansion = [x]
                        expansion_map[x.name()] = expansion

                    expanded_atomic_impulses.append(expansion_map[x.name()])

                new_impulses = [ImpulseInteraction(x, ops=self.impulse.ops) for x in itertools.product(*expanded_atomic_impulses)]

            elif type(self.impulse).__name__ == 'NNImpulse':
                expanded_atomic_impulses = []
                for x in self.impulse.impulses():
                    if x.name() not in expansion_map:
                        if x.categorical(X):
                            found = False
                            for _X in X:
                                if x.id in _X:
                                    found = True
                                    vals = sorted(_X[x.id].unique())[1:]
                                    expansion = [Impulse('_'.join([x.id, pythonize_string(str(val))]), ops=x.ops, is_re=x.is_re) for val in vals]
                                    break
                            assert found, 'Impulse %s not found in data.' % x.id
                        else:
                            expansion = [x]
                        expansion_map[x.name()] = expansion

                    expanded_atomic_impulses.append(expansion_map[x.name()])

                new_impulses = [
                    NNImpulse(
                        sum(expanded_atomic_impulses, []),
                        impulses_as_inputs=self.impulse.impulses_as_inputs,
                        inputs_to_add=self.impulse.inputs_added,
                        inputs_to_drop=self.impulse.inputs_dropped,
                        nn_config=self.impulse.nn_config
                    )
                ]

            else:
                if not self.impulse.name() in expansion_map and not isinstance(self.impulse, NNImpulse):
                    if self.impulse.categorical(X):
                        found = False
                        for _X in X:
                            if self.impulse.id in _X:
                                found = True
                                vals = sorted(_X[self.impulse.id].unique())[1:]
                                expansion = [Impulse('_'.join([self.impulse.id, pythonize_string(str(val))]), ops=self.impulse.ops, is_re=self.impulse.is_re) for val in vals]
                                break
                        assert found, 'Impulse %s not found in data.' % self.impulse.id
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
                    fixed=self.fixed,
                    rangf=self.rangf[:],
                    nn_impulses=self.nn_impulses,
                    nn_config=self.nn_config,
                    impulses_as_inputs=self.impulses_as_inputs,
                    inputs_to_add=self.inputs_added,
                    inputs_to_drop=self.inputs_dropped,
                    param_init=self.param_init,
                    trainable=self.trainable,
                    response_params_list=self.response_params_list
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
                    nn_impulses=self.nn_impulses,
                    nn_config=self.nn_config,
                    impulses_as_inputs=self.impulses_as_inputs,
                    inputs_to_add=self.inputs_added,
                    inputs_to_drop=self.inputs_dropped,
                    param_init=self.param_init,
                    trainable=self.trainable,
                    response_params_list=self.response_params_list
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
                    elif isinstance(response, NNImpulse):
                        expanded_interaction_impulse = []
                        for impulse in response.impulses():
                            expansion = expansion_map[impulse.name()]
                            expanded_interaction_impulse.append(expansion)
                        expanded_interaction_impulse = [
                            NNImpulse(
                                sum(expanded_interaction_impulse, []),
                                impulses_as_inputs=response.impulses_as_inputs,
                                inputs_to_add=response.inputs_added,
                                inputs_to_drop=response.inputs_dropped,
                                nn_config=response.nn_config
                            )
                        ]
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

    def re_transform(self, X, expansion_map=None):
        """
        Generate transformed copy of node with regex-matching predictors in **X** expanded.
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
                            if response.is_re:
                                m = response.get_matcher()
                                expansion = []
                                for _X in X:
                                    for col in _X:
                                        if m.match(col):
                                            expansion.append(
                                                Impulse(
                                                    col,
                                                    ops=response.ops,
                                                    is_re=False
                                                )
                                            )
                                assert expansion, 'Impulse %d not found in data.' % response.id
                            else:
                                expansion = [response]
                            expansion_map[response.name()] = expansion
                    elif isinstance(response, ImpulseInteraction):
                        for subresponse in response.impulses():
                            if not subresponse.name() in expansion_map:
                                if subresponse.is_re:
                                    m = subresponse.get_matcher()
                                    expansion = []
                                    for _X in X:
                                        for col in _X:
                                            if m.match(col):
                                                expansion.append(
                                                    Impulse(
                                                        col,
                                                        ops=subresponse.ops,
                                                        is_re=False
                                                    )
                                                )
                                    assert expansion, 'Impulse %d not found in data.' % subresponse.id
                                else:
                                    expansion = [subresponse]
                                expansion_map[subresponse.name()] = expansion

            if type(self.impulse).__name__ == 'ImpulseInteraction':
                expanded_atomic_impulses = []
                for x in self.impulse.impulses():
                    if x.name() not in expansion_map:
                        if x.is_re:
                            m = x.get_matcher()
                            expansion = []
                            for _X in X:
                                for col in _X:
                                    if m.match(col):
                                        expansion.append(
                                            Impulse(
                                                col,
                                                ops=x.ops,
                                                is_re=False
                                            )
                                        )
                            assert expansion, 'Impulse %s not found in data.' % x.id
                        else:
                            expansion = [x]
                        expansion_map[x.name()] = expansion

                    expanded_atomic_impulses.append(expansion_map[x.name()])

                new_impulses = [ImpulseInteraction(x, ops=self.impulse.ops) for x in itertools.product(*expanded_atomic_impulses)]

            elif type(self.impulse).__name__ == 'NNImpulse':
                expanded_atomic_impulses = []
                for x in self.impulse.impulses():
                    if x.name() not in expansion_map:
                        if x.is_re:
                            m = x.get_matcher()
                            expansion = []
                            for _X in X:
                                for col in _X:
                                    if m.match(col):
                                        expansion.append(
                                            Impulse(
                                                col,
                                                ops=x.ops,
                                                is_re=False
                                            )
                                        )
                            assert expansion, 'Impulse %s not found in data.' % x.id
                        else:
                            expansion = [x]
                        expansion_map[x.name()] = expansion

                    expanded_atomic_impulses.append(expansion_map[x.name()])

                new_impulses = [
                    NNImpulse(
                        sum(expanded_atomic_impulses, []),
                        impulses_as_inputs=self.impulse.impulses_as_inputs,
                        inputs_to_add=self.impulse.inputs_added,
                        inputs_to_drop=self.impulse.inputs_dropped,
                        nn_config=self.impulse.nn_config
                    )
                ]

            else:
                if not self.impulse.name() in expansion_map and not isinstance(self.impulse, NNImpulse):
                    if self.impulse.is_re:
                        m = self.impulse.get_matcher()
                        expansion = []
                        for _X in X:
                            for col in _X:
                                if m.match(col):
                                    expansion.append(
                                        Impulse(
                                            col,
                                            ops=self.impulse.ops,
                                            is_re=False
                                        )
                                    )
                        assert expansion, 'Impulse %s not found in data.' % self.impulse.id
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
                    fixed=self.fixed,
                    rangf=self.rangf[:],
                    nn_impulses=self.nn_impulses,
                    nn_config=self.nn_config,
                    impulses_as_inputs=self.impulses_as_inputs,
                    inputs_to_add=self.inputs_added,
                    inputs_to_drop=self.inputs_dropped,
                    param_init=self.param_init,
                    trainable=self.trainable,
                    response_params_list=self.response_params_list
                )
                irf_expansion.append(new_irf)

                self_transformed.append(new_irf)

            expansion_map[self.name()] = irf_expansion

        elif self.family is None:
            ## ROOT node
            children = []
            for c in self.children:
                c_children = [x for x in c.re_transform(X, expansion_map=expansion_map)]
                children += c_children
            new_irf = IRFNode()
            for c in children:
                new_irf.add_child(c)
            self_transformed.append(new_irf)

        else:
            children = []
            for c in self.children:
                c_children = [x for x in c.re_transform(X, expansion_map=expansion_map)]
                children += c_children
            for c in children:
                new_irf = IRFNode(
                    family=self.family,
                    irfID=self.irfID,
                    fixed=self.fixed,
                    rangf=self.rangf,
                    nn_impulses=self.nn_impulses,
                    nn_config=self.nn_config,
                    impulses_as_inputs=self.impulses_as_inputs,
                    inputs_to_add=self.inputs_added,
                    inputs_to_drop=self.inputs_dropped,
                    param_init=self.param_init,
                    trainable=self.trainable,
                    response_params_list=self.response_params_list
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
                    elif isinstance(response, NNImpulse):
                        expanded_interaction_impulse = []
                        for impulse in response.impulses():
                            expansion = expansion_map[impulse.name()]
                            expanded_interaction_impulse.append(expansion)
                        expanded_interaction_impulse = [
                            NNImpulse(
                                sum(expanded_interaction_impulse, []),
                                impulses_as_inputs=response.impulses_as_inputs,
                                inputs_to_add=response.inputs_added,
                                inputs_to_drop=response.inputs_dropped,
                                nn_config=response.nn_config
                            )
                        ]
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

        :return: ``dict``; map from random grouping factors to data structure representing formula terms.
            Data structure contains 2 fields, ``'impulses'`` containing impulses and ``'irf'`` containing IRF Nodes.
        """

        if self.terminal():
            out = {}
            if self.fixed:
                out[None] = [{
                    'impulses': self.impulses(include_nn=True, include_nn_inputs=False),
                    'irf': '',
                    'nn_key': None
                }]
            for rangf in self.rangf:
                out[rangf] = [{
                    'impulses': self.impulses(include_nn=True, include_nn_inputs=False),
                    'irf': '',
                    'nn_key': None
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
                    term['nn_key'] = self.nn_key

        return out

    def __str__(self):
        s = self.name()
        if len(self.rangf) > 0:
            s += '; rangf: ' + ','.join(self.rangf)
        if len(self.trainable) > 0:
            s +=  '; trainable params: ' + ', '.join(self.trainable)
        if self.response_params:
            vals = []
            for key in self.response_params:
                if key is None:
                    val = 'any: ' + ', '.join(sorted(self.response_params[key]))
                else:
                    val = key + ': ' + ', '.join([x for x in Formula.RESPONSE_DISTRIBUTIONS[key]['params'] \
                                                  if x in self.response_params[key]])
                vals.append(val)
            s += '; ' + '; '.join(vals)
        if self.family == 'NN':
            for key in self.nn_config:
                s += '; %s: %s' % (key, self.nn_config[key])
        indent = '  '
        for c in self.children:
            s += '\n%s' % indent + str(c).replace('\n', '\n%s' % indent)
        return s

