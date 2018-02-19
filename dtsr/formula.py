import sys
import re
import ast
import itertools
import numpy as np

interact = re.compile('([^ ]+):([^ ]+)')

def z(df):
    # sys.stderr.write('SD of %s:\n'%df.name)
    # sys.stderr.write(str(df.std(axis=0)) + '\n')
    return (df-df.mean(axis=0))/df.std(axis=0)

def c(df):
    return df-df.mean(axis=0)

def s(df):
    # sys.stderr.write('SD of %s:\n'%df.name)
    # sys.stderr.write(str(df.std(axis=0)) + '\n')
    return df/df.std(axis=0)

class Formula(object):
    """
    A class for parsing R-style mixed-effects model formula strings and applying them to DTSR design matrices.
    
    # Arguments
        bform_str: String. An R-style mixed-effects model formula string
    """
    def __init__(self, bform_str):
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
                    if type(x).__name__ == 'IRFTreeNode':
                        raise ValueError('Interaction terms may not dominate IRF terms in DTSR formula strings')
                new = InteractionImpulse(terms=subterms, ops=ops)
                terms.append(new)
            elif type(t.op).__name__ == 'Mult':
                assert len(ops) == 0, 'Transformation of term expansions is not supported in DTSR formula strings'
                subterms = []
                self.process_ast(t.left, terms=subterms, has_intercept=has_intercept, rangf=rangf)
                self.process_ast(t.right, terms=subterms, has_intercept=has_intercept, rangf=rangf)
                for x in subterms:
                    if type(x).__name__ == 'IRFTreeNode':
                        raise ValueError('Term expansions may not dominate IRF terms in DTSR formula strings')
                new = InteractionImpulse(terms=subterms, ops=ops)
                terms += subterms
                terms.append(new)
            elif type(t.op).__name__ == 'Pow':
                assert len(ops) == 0, 'Transformation of term expansions is not supported in DTSR formula strings'
                subterms = []
                self.process_ast(t.left, terms=subterms, has_intercept=has_intercept, rangf=rangf)
                for x in subterms:
                    if type(x).__name__ == 'IRFTreeNode':
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
            elif t.func.id in Formula.IRF:
                raise ValueError('IRF calls can only occur as inputs to C() in DTSR formula strings')
            else:
                assert len(t.args) <= 1, 'Only unary ops on variables supported in DTSR formula strings'
                subterms = []
                self.process_ast(t.args[0], terms=subterms, has_intercept=has_intercept, ops=ops + [t.func.id], rangf=rangf)
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
        assert t.func.id in Formula.IRF, 'Ill-formed model string: process_irf() called on non-IRF node'
        irf_id = None
        coef_id = None
        cont=False
        ranirf=False
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

        if isinstance(input, IRFNode):
            new = IRFNode(
                family=t.func.id,
                irfID=irf_id,
                ops=ops,
                fixed=rangf is None,
                rangf=rangf if ranirf else None
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
                rangf=rangf
            )

            p = self.process_irf(
                t,
                input=new,
                rangf=rangf
            )

        return new

    def apply_op(self, op, arr):
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
                # else:
                #     raise ValueError('Unrecognized term "%s" in model formula' %term.id)
            else:
                df[impulse.name()] = df[impulse.id]
            for i in range(len(ops)):
                op = ops[i]
                df[impulse.name()] = self.apply_op(op, df[impulse.name()])
        return df

    def apply_formula(self, y, X):
        if self.dv not in y.columns:
            y = self.apply_ops(self.dv_term, y)
        impulses = self.t.impulses()
        for t in impulses:
            X = self.apply_ops(t, X)
        return y, X

    IRF = [
        'DiracDelta',
        'Exp',
        'SteepExp',
        'Gamma',
        'ShiftedGamma',
        'GammaKgt1',
        'ShiftedGammaKgt1',
        'Normal',
        'SkewNormal',
        'EMG',
        'BetaPrime',
        'ShiftedBetaPrime',
    ]

    def __str__(self):
        return self.bform_str

class Impulse(object):
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
            rangf=None
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
        else:
            self.ops = [] if ops is None else ops[:]
            self.cont = cont
            self.impulse = impulse
            self.family = family
            self.irfID = irfID
            self.coefID = coefID
            self.fixed = fixed
            self.rangf = [] if rangf is None else rangf if isinstance(rangf, list) else [rangf]

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

    def __str__(self):
        s = self.name()
        if len(self.rangf) > 0:
            s += '; rangf: ' + ','.join(self.rangf)
        indent = '  '
        for c in self.children:
            s += '\n%s' % indent + str(c).replace('\n', '\n%s' % indent)
        return s
