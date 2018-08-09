import sys
import re
import math
import ast
import itertools
import numpy as np

from .data import z, c, s, compute_time_mask, expand_history
from .util import names2ix, sn

interact = re.compile('([^ ]+):([^ ]+)')
spillover = re.compile('^.+S[0-9]+$')
split_irf = re.compile('(.+)\(([^(]+)')
spline = re.compile('S((o([0-9]+))?(b([0-9]+))?(l([0-9]+))?(p([0-9]+))?(i([0-1]))?)?$')


class Formula(object):
    """
    A class for parsing R-style mixed-effects model formula strings and applying them to DTSR design matrices.
    
    # Arguments
        bform_str: String. An R-style mixed-effects model formula string
    """

    IRF = [
        'DiracDelta',
        'Exp',
        'ExpRateGT1',
        'Gamma',
        'ShiftedGamma',
        'GammaKgt1',
        'GammaShapeGT1',
        'ShiftedGammaKgt1',
        'ShiftedGammaShapeGT1',
        'Normal',
        'SkewNormal',
        'EMG',
        'BetaPrime',
        'ShiftedBetaPrime',
        'HRFSingleGamma',
        'HRFDoubleGamma',
        'HRFDoubleGammaUnconstrained'
    ]

    IRF_PARAMS = {
        'DiracDelta': [],
        'Exp': ['beta'],
        'ExpRateGT1': ['beta'],
        'Gamma': ['alpha', 'beta'],
        'ShiftedGamma': ['alpha', 'beta', 'delta'],
        'GammaKgt1': ['alpha', 'beta'],
        'GammaShapeGT1': ['alpha', 'beta'],
        'ShiftedGammaKgt1': ['alpha', 'beta', 'delta'],
        'ShiftedGammaShapeGT1': ['alpha', 'beta', 'delta'],
        'Normal': ['mu', 'sigma'],
        'SkewNormal': ['mu', 'sigma', 'alpha'],
        'EMG': ['mu', 'sigma', 'beta'],
        'BetaPrime': ['alpha', 'beta'],
        'ShiftedBetaPrime': ['alpha', 'beta', 'delta'],
        'HRFSingleGamma': ['alpha', 'beta'],
        'HRFDoubleGamma': ['alpha_main', 'beta', 'alpha_undershoot_offset', 'c'],
        'HRFDoubleGammaUnconstrained': ['alpha_main', 'beta_main', 'alpha_undershoot', 'beta_undershoot', 'c']
    }

    SPLINE_DEFAULT_ORDER = 2
    SPLINE_DEFAULT_BASES = 10
    SPLINE_DEFAULT_ROUGHNESS_PENALTY = 0.001
    SPLINE_DEFAULT_SPACING_POWER = 1
    SPLINE_DEFAULT_INSTANTANEOUS = True

    @staticmethod
    def irf_params(family):
        if family in Formula.IRF_PARAMS:
            out = Formula.IRF_PARAMS[family]
        elif Formula.is_spline(family):
            bs = Formula.bases(family)
            instantaneous = Formula.instantaneous(family)
            out = ['x%s' % i for i in range(2, bs + 1)] + ['y%s' % i for i in range(1 + (1-instantaneous), bs)]
        else:
            out = []

        return out

    @staticmethod
    def is_spline(family):
        if family is None:
            out = False
        else:
            out = spline.match(family) is not None
        return out

    @staticmethod
    def order(family):
        if family is None:
            out = None
        else:
            if Formula.is_spline(family):
                order = spline.match(family).group(3)
                if order is None:
                    order = Formula.SPLINE_DEFAULT_ORDER
                out = int(order)
            else:
                out = None
        return out

    @staticmethod
    def bases(family):
        if family is None:
            out = None
        else:
            if Formula.is_spline(family):
                bases = spline.match(family).group(5)
                if bases is None:
                    bases = Formula.SPLINE_DEFAULT_BASES
                out = int(bases)
            else:
                out = None
        return out

    @staticmethod
    def roughness_penalty(family):
        if family is None:
            out = None
        else:
            if Formula.is_spline(family):
                roughness_penalty = spline.match(family).group(7)
                if roughness_penalty is None:
                    out = Formula.SPLINE_DEFAULT_ROUGHNESS_PENALTY
                else:
                    out = float('0.' + roughness_penalty)
            else:
                out = None
        return out

    @staticmethod
    def spacing_power(family):
        if family is None:
            out = None
        else:
            if Formula.is_spline(family):
                spacing_power = spline.match(family).group(9)
                if spacing_power is None:
                    spacing_power = Formula.SPLINE_DEFAULT_SPACING_POWER
                out = int(spacing_power)
            else:
                out = None
        return out

    @staticmethod
    def instantaneous(family):
        if family is None:
            out = None
        else:
            if Formula.is_spline(family):
                instantaneous = spline.match(family).group(11)
                if instantaneous is None:
                    instantaneous = Formula.SPLINE_DEFAULT_BASES
                out = bool(int(instantaneous))
            else:
                out = None
        return out

    def __init__(self, bform_str):
        self.build(bform_str)

    def build(self, bform_str):
        self.bform_str = bform_str

        lhs, rhs = bform_str.strip().split('~')

        dv = ast.parse(lhs.strip().replace('.(', '(').replace(':', '%'))
        dv_term = []
        self.process_ast(dv.body[0].value, terms=dv_term)
        self.dv_term = dv_term[0]
        self.dv = self.dv_term.name()

        rhs = ast.parse(rhs.strip().replace('.(', '(').replace(':', '%'))

        self.t = IRFNode()
        terms = []
        self.has_intercept = {None: True}
        self.process_ast(rhs.body[0].value, terms=terms, has_intercept=self.has_intercept)

        self.rangf = sorted([x for x in list(self.has_intercept.keys()) if x is not None])

    def process_ast(
            self,
            t,
            terms=None,
            has_intercept=None,
            ops=None,
            rangf=None
    ):
        if terms is None:
            terms = []
        if has_intercept is None:
            has_intercept = {}
        if rangf not in has_intercept:
            has_intercept[rangf] = True
        if ops is None:
            ops = []
        if type(t).__name__ == 'BinOp':
            if type(t.op).__name__ == 'Add':
                assert len(ops) == 0, 'Transformation of multiple terms is not supported in DTSR formula strings'
                self.process_ast(t.left, terms=terms, has_intercept=has_intercept, ops=ops, rangf=rangf)
                self.process_ast(t.right, terms=terms, has_intercept=has_intercept, ops=ops, rangf=rangf)
            elif type(t.op).__name__ == 'BitOr':
                assert len(ops) == 0, 'Transformation of random terms is not supported in DTSR formula strings'
                assert rangf is None, 'Random terms may not be embedded under other random terms in DTSR formula strings'
                self.process_ast(t.left, has_intercept=has_intercept, rangf=t.right.id)
            elif type(t.op).__name__ == 'Mod':
                subterms = []
                self.process_ast(t.left, terms=subterms, has_intercept=has_intercept, rangf=rangf)
                self.process_ast(t.right, terms=subterms, has_intercept=has_intercept, rangf=rangf)
                for x in subterms:
                    if type(x).__name__ == 'IRFNode':
                        raise ValueError('Interaction terms may not dominate IRF terms in DTSR formula strings')
                new = InteractionImpulse(terms=subterms, ops=ops)
                terms.append(new)
            elif type(t.op).__name__ == 'Mult':
                assert len(ops) == 0, 'Transformation of term expansions is not supported in DTSR formula strings'
                subterms = []
                self.process_ast(t.left, terms=subterms, has_intercept=has_intercept, rangf=rangf)
                self.process_ast(t.right, terms=subterms, has_intercept=has_intercept, rangf=rangf)
                for x in subterms:
                    if type(x).__name__ == 'IRFNode':
                        raise ValueError('Term expansions may not dominate IRF terms in DTSR formula strings')
                new = InteractionImpulse(terms=subterms, ops=ops)
                terms += subterms
                terms.append(new)
            elif type(t.op).__name__ == 'Pow':
                assert len(ops) == 0, 'Transformation of term expansions is not supported in DTSR formula strings'
                subterms = []
                self.process_ast(t.left, terms=subterms, has_intercept=has_intercept, rangf=rangf)
                for x in subterms:
                    if type(x).__name__ == 'IRFNode':
                        raise ValueError('Term expansions may not dominate IRF terms in DTSR formula strings')
                order = min(int(t.right.n), len(subterms))
                for i in range(1, order + 1):
                    collections = itertools.combinations(subterms, i)
                    for tup in collections:
                        if i > 1:
                            new = InteractionImpulse(list(tup), ops=ops)
                            terms.append(new)
                        else:
                            terms.append(tup[0])
        elif type(t).__name__ == 'Call':
            if t.func.id == 'C':
                assert len(t.args) == 2, 'C() takes exactly two arguments in DTSR formula strings'
                subterms = []
                self.process_ast(t.args[0], terms=subterms, has_intercept=has_intercept, rangf=rangf)
                for x in subterms:
                    new = self.process_irf(t.args[1], input=x, ops=None, rangf=rangf)
                    terms.append(new)
            elif t.func.id in Formula.IRF or spline.match(t.func.id) is not None:
                raise ValueError('IRF calls can only occur as inputs to C() in DTSR formula strings')
            else:
                assert len(t.args) <= 1, 'Only unary ops on variables supported in DTSR formula strings'
                subterms = []
                self.process_ast(t.args[0], terms=subterms, has_intercept=has_intercept, ops=[t.func.id] + ops, rangf=rangf)
                terms += subterms
        elif type(t).__name__ == 'Name':
            new = Impulse(t.id, ops=ops)
            terms.append(new)
        elif type(t).__name__ == 'NameConstant':
            new = Impulse(t.value, ops=ops)
            terms.append(new)
        elif type(t).__name__ == 'Num':
            new = Impulse(str(t.n), ops=ops)
            terms.append(new)
        else:
            raise ValueError('Operation "%s" is not supported in DTSR formula strings' %type(t).__name___)

        for t in terms:
            if t.name() == '0':
                has_intercept[rangf] = False

    def process_irf(
            self,
            t,
            input,
            ops=None,
            rangf=None
    ):
        if ops is None:
            ops = []
        assert t.func.id in Formula.IRF or spline.match(t.func.id) is not None, 'Ill-formed model string: process_irf() called on non-IRF node'
        irf_id = None
        coef_id = None
        cont = False
        ranirf = False
        trainable = None
        param_init={}
        order = None
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

        return new

    def apply_op(self, op, arr):
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

    def apply_ops(self, impulse, df):
        ops = impulse.ops
        if impulse.name() not in df.columns:
            if impulse.id not in df.columns:
                if type(impulse).__name__ == 'InteractionTerm':
                    for t in impulse.terms:
                        df = self.apply_ops(t, df)
                    df[impulse.id] = df[[x.name() for x in impulse.terms]].product(axis=1)
                else:
                    raise ValueError('Unrecognized term "%s" in model formula' %impulse.id)
            else:
                df[impulse.name()] = df[impulse.id]
            for i in range(len(ops)):
                op = ops[i]
                df[impulse.name()] = self.apply_op(op, df[impulse.name()])
        return df

    def compute_2d_predictor(
            self,
            predictor_name,
            X,
            first_obs,
            last_obs,
            history_length=128,
            minibatch_size=50000
    ):
        supported = [
            'cosdist2D',
            'eucldist2D'
        ]

        assert predictor_name in supported, '2D predictor "%s" not currently supported' %predictor_name

        if predictor_name in ['cosdist2D', 'eucldist2D']:
            is_embedding_dimension = re.compile('d([0-9]+)')

            embedding_colnames = [c for c in X.columns if is_embedding_dimension.match(c)]
            embedding_colnames = sorted(embedding_colnames, key=lambda x: float(x[1:]))

            assert len(embedding_colnames) > 0, 'Model formula contains vector distance predictors but no embedding columns found in the input data'

            new_2d_predictor = []
            if predictor_name == 'cosdist2D':
                new_2d_predictor_name = 'cosdist2D'
                sys.stderr.write('Computing pointwise cosine distances...\n')
            elif predictor_name == 'eucldist2D':
                sys.stderr.write('Computing pointwise Euclidean distances...\n')
                new_2d_predictor_name = 'eucldist2D'

            for i in range(0, len(first_obs), minibatch_size):
                sys.stderr.write('\rProcessing batch %d/%d' %(i/minibatch_size + 1, math.ceil(len(first_obs)/minibatch_size)))
                sys.stderr.flush()
                X_embeddings, _, _ = expand_history(
                    np.array(X[embedding_colnames]),
                    np.array(X['time']),
                    np.array(first_obs)[i:i+minibatch_size],
                    np.array(last_obs)[i:i+minibatch_size],
                    history_length,
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

            sys.stderr.write('\n')
            new_2d_predictor = np.concatenate(new_2d_predictor, axis=0)

        return new_2d_predictor_name, new_2d_predictor

    def apply_op_2d(self, op, arr, time_mask):
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

    def apply_ops_2d(self, impulse, X_2d_predictor_names, X_2d_predictors, time_mask, history_length=128):
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
            y,
            X_response_aligned_predictor_names=None,
            X_response_aligned_predictors=None,
            X_2d_predictor_names=None,
            X_2d_predictors=None,
            history_length=128
    ):
        if self.dv not in y.columns:
            y = self.apply_ops(self.dv_term, y)
        impulses = self.t.impulses()

        if X_2d_predictor_names is None:
            X_2d_predictor_names = []
        if X_response_aligned_predictor_names is None:
            X_response_aligned_predictor_names = []

        time_mask = None

        for impulse in impulses:
            if impulse.is_2d:
                if time_mask is None:
                    time_mask = compute_time_mask(
                        X.time,
                        y.first_obs,
                        y.last_obs,
                        history_length=history_length
                    )

                if impulse.id not in X_2d_predictor_names:
                    new_2d_predictor_name, new_2d_predictor = self.compute_2d_predictor(
                        impulse.id,
                        X,
                        y.first_obs,
                        y.last_obs,
                        history_length=history_length
                    )
                    X_2d_predictor_names.append(new_2d_predictor_name)
                    if X_2d_predictors is None:
                        X_2d_predictors = new_2d_predictor
                    else:
                        X_2d_predictors = np.concatenate([X_2d_predictors, new_2d_predictor], axis=2)

                X_2d_predictor_names, X_2d_predictors = self.apply_ops_2d(
                    impulse,
                    X_2d_predictor_names,
                    X_2d_predictors,
                    time_mask,
                    history_length=history_length
                )

            elif impulse.id not in X.columns:
                if impulse.name() not in X_response_aligned_predictor_names:
                    X_response_aligned_predictor_names.append(impulse.name())
                    if X_response_aligned_predictors is None:
                        X_response_aligned_predictors = y[[impulse.id]]
                    else:
                        X_response_aligned_predictors[[impulse.id]] = y[[impulse.id]]
                    X_response_aligned_predictors = self.apply_ops(impulse, X_response_aligned_predictors)
            else:
                X = self.apply_ops(impulse, X)
        for col in [x for x in X.columns if spillover.match(x)]:
            X[col] = X[col].fillna(0)
        return X, y, X_response_aligned_predictor_names, X_response_aligned_predictors, X_2d_predictor_names, X_2d_predictors

    def ablate_impulses(self, impulse_ids):
        if not isinstance(impulse_ids, list):
            impulse_ids = [impulse_ids]
        self.t.ablate_impulses(impulse_ids)

    def unablate_impulses(self, impulse_ids):
        if not isinstance(impulse_ids, list):
            impulse_ids = [impulse_ids]
        self.t.unablate_impulses(impulse_ids)

    def insert_impulses(self, impulses, irf_str, rangf=['subject']):
        if not isinstance(impulses, list):
            impulses = [impulses]

        bform = str(self)
        bform += ' + C(' + ' + '.join(impulses) + ', ' + irf_str + ')'
        for gf in rangf:
            bform += ' + (C(' + ' + '.join(impulses) + ', ' + irf_str + ') | ' + gf + ')'

        self.build(bform)

    def remove_impulses(self, impulse_ids):
        if not isinstance(impulse_ids, list):
            impulse_ids = [impulse_ids]
        self.t.remove_impulses(impulse_ids)

    def __str__(self):
        out = str(self.dv_term) + ' ~ '

        if not self.has_intercept[None]:
            out += '0 + '

        terms = self.t.formula_terms()
        term_strings = []

        if None in terms:
            fixed = terms.pop(None)
            new_terms = {}
            for term in fixed:
                if term['irf'] in new_terms:
                    new_terms[term['irf']]['impulses'] += term['impulses']
                else:
                    new_terms[term['irf']] = term
            new_terms = [new_terms[x] for x in new_terms]
            term_strings.append(' + '.join(['C(%s, %s)' %(' + '.join([x.name() for x in y['impulses']]), y['irf']) for y in new_terms]))

        for rangf in terms:
            ran = terms[rangf]
            new_terms = {}
            for term in ran:
                if term['irf'] in new_terms:
                    new_terms[term['irf']]['impulses'] += term['impulses']
                else:
                    new_terms[term['irf']] = term
            new_terms = [new_terms[x] for x in new_terms]
            new_terms_str = '('
            if not self.has_intercept[rangf]:
                new_terms_str += '0 + '
            new_terms_str += ' + '.join(['C(%s, %s)' % (' + '.join([x.name() for x in y['impulses']]), y['irf']) for y in new_terms]) + ' | %s)' %rangf
            term_strings.append(new_terms_str)

        out += ' + '.join(term_strings)

        for key in self.has_intercept:
            if key is not None and not key in terms and self.has_intercept[key]:
                out += ' + (1 | %s)' %key

        return out

    def to_lmer_formula_string(self, z=False):
        fixed = []
        random = {}

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

        out = str(self.dv_term) + ' ~ '

        if not self.has_intercept[None]:
            out += '0 + '

        out += ' + '.join([x for x in fixed])
        for gf in random:
            out += ' + (' + ('1 + ' if self.has_intercept[gf] else '0 + ') + ' + '.join([x for x in random[gf]]) + ' | ' + gf + ')'

        return out

class Impulse(object):
    def __init__(self, name, ops=None):
        if ops is None:
            ops = []
        self.ops = ops[:]
        self.name_str = name
        for op in self.ops:
            self.name_str = op + '(' + self.name_str + ')'
        self.id = name
        self.is_2d = name.endswith('2D')

    def __str__(self):
        return self.name_str

    def name(self):
        return self.name_str

class InteractionImpulse(object):
    def __init__(self, terms, ops=None):
        if ops is None:
            ops = []
        self.ops = ops[:]
        self.terms = []
        names = set()
        for t in terms:
            if t.name() not in names:
                names.add(t.name())
                self.terms.append(t)
        self.name_str = ':'.join([t.name() for t in terms])
        for op in self.ops:
            self.name_str = op + '(' + self.name_str + ')'
        self.id = ':'.join([x.name() for x in self.terms])

    def __str__(self):
        return self.name_str

    def name(self):
        return self.name_str

class IRFNode(object):
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
            self.rangf = [] if rangf is None else rangf if isinstance(rangf, list) else [rangf]

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

    def add_child(self, t):
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
        if not isinstance(rangf, list):
            rangf = [rangf]
        for gf in rangf:
            if gf not in self.rangf:
                self.rangf.append(gf)

    def local_name(self):
        if self.irfID is None:
            out = '.'.join([self.family] + self.impulse_names())
        else:
            out = self.irfID
        return out

    def name(self):
        if self.family is None:
            return 'ROOT'
        if self.p is None or self.p.name() == 'ROOT':
            p_name = ''
        else:
            p_name = self.p.name() + '-'
        return p_name + self.local_name()

    def irf_id(self):
        if not self.terminal():
            if self.irfID is None:
                out = self.name()
            else:
                out = self.irfID
            return out
        return None

    def coef_id(self):
        if self.terminal():
            if self.coefID is None:
                return self.name()
            return self.coefID
        return None

    def terminal(self):
        return self.family == 'Terminal'

    def depth(self):
        d = 1
        for c in self.children:
            if c.depth() + 1 > d:
                d = c.depth() + 1
        return d

    def has_composed_irf(self):
        return self.depth() > 3

    def is_spline(self):
        return Formula.is_spline(self.family)

    def order(self):
        return Formula.order(self.family)

    def bases(self):
        return Formula.bases(self.family)

    def spacing_power(self):
        return Formula.spacing_power(self.family)

    def roughness_penalty(self):
        return Formula.roughness_penalty(self.family)

    def instantaneous(self):
        return Formula.instantaneous(self.family)

    def impulses(self):
        out = []
        if self.terminal():
            out.append(self.impulse)
        else:
            for c in self.children:
                for imp in c.impulses():
                    if imp.name() not in [x.name() for x in out]:
                        out.append(imp)
        return out

    def impulse_names(self):
        return [x.name() for x in self.impulses()]

    def impulses_by_name(self):
        out = {}
        for x in self.impulses():
            out[x.name()] = x
        return out

    def terminals(self):
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
        return [x.name() for x in self.terminals()]

    def terminals_by_name(self):
        out = {}
        for x in self.terminals():
            out[x.name()] = x
        return out

    def coef_names(self):
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
        out = []
        if self.terminal():
            if self.fixed:
                out.append(self.coef_id())
        else:
            for c in self.children:
                names = c.fixed_coef_names()
                for name in names:
                    if name not in out:
                        out.append(name)
        return out

    def atomic_irf_by_family(self):
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

    def coef2impulse(self):
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
        out = {}
        if self.terminal():
            for gf in self.rangf:
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

    def irf_by_rangf(self):
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
        out = {self.name(): self}
        for c in self.children:
            nt = c.node_table()
            for x in nt:
                assert not x in out, 'Duplicate node ids appear in IRF tree'
                out[x] = nt[x]
        return out

    def pc_transform(self, n_pc, pointers=None):
        self_transformed = []

        if self.terminal():
            if self.impulse.name() == 'rate':
                self_pc = IRFNode(
                    family='Terminal',
                    impulse=self.impulse,
                    coefID=self.coefID,
                    cont=self.cont,
                    fixed=self.fixed,
                    rangf=self.rangf[:]
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
                        rangf=self.rangf[:]
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
                    rangf=self.rangf
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

    @staticmethod
    def pointers2namemmaps(p):
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
        if not isinstance(impulse_ids, list):
            impulse_ids = [impulse_ids]
        if self.terminal():
            if self.impulse.id in impulse_ids:
                self.fixed = False
        else:
            for c in self.children:
                c.ablate_impulses(impulse_ids)

    def unablate_impulses(self, impulse_ids):
        if not isinstance(impulse_ids, list):
            impulse_ids = [impulse_ids]
        if self.terminal():
            if self.impulse.id in impulse_ids:
                self.fixed = True
        else:
            for c in self.children:
                c.unablate_impulses(impulse_ids)

    def remove_impulses(self, impulse_ids):
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
                    outer = None
                    if term['irf'] != '':
                        outer = split_irf.match(term['irf']).groups()
                    inner = []
                    if self.irfID is not None:
                        inner.append('irf_id=%s' %self.irfID)
                    if self.coefID is not None:
                        inner.append('coef_id=%s' %self.coefID)
                    if key in self.rangf:
                        inner.append('ran=T')
                    if self.cont:
                        inner.append('cont=T')
                    if len(self.param_init) > 0:
                        inner.append(', '.join(['%s=%s' %(x, self.param_init[x]) for x in self.param_init]))
                    if set(self.trainable) != set(Formula.irf_params(self.family)):
                        inner.append('trainable=%s' %self.trainable)
                    new_irf = self.family + '(' + ', '.join(inner) + ')'
                    if outer is not None:
                        new_irf = outer[0] + '(' + new_irf
                        if outer[1].startswith(')'):
                            new_irf += outer[1]
                        else:
                            new_irf += ', ' + outer[1]
                    term['irf'] = new_irf

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
