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

        self.terminals = {}
        self.preterminal_names = []
        self.coefficient_names = []
        self.fixed_coefficient_names = []
        self.irf_tree = IRFTreeNode()
        self.atomic_irf_by_family = {}
        self.random = {}

        dv = ast.parse(lhs.strip().replace('.(', '(').replace(':','%'))
        self.dv_term = self.process_ast(dv.body[0].value)[0]
        self.dv = self.dv_term.name()

        rhs = ast.parse(rhs.strip().replace('.(', '(').replace(':','%'))
        terms = self.process_ast(rhs.body[0].value)

        self.intercept = True
        for t in terms:
            if t.name() == '0':
                self.intercept = False
            elif t.name() == '1':
                self.intercept = True
            elif not type(t).__name__ in ['IRFTreeNode', 'RandomTerm']:
                raise ValueError('All top-level terms in a DTSR model must be either IRF terms or random terms.')

        self.terminal_names = sorted(self.terminals.keys())
        self.preterminal_names = sorted(self.preterminal_names)
        self.coefficient_names = sorted(self.coefficient_names)
        self.fixed_coefficient_names = sorted(self.fixed_coefficient_names)
        self.ran_names = sorted(self.random.keys())
        self.rangf = [self.random[r].gf for r in self.ran_names]


    def process_ast(self, t, ops=None, under_IRF=False, random_coefficients=None):
        if ops is None:
            ops = []
        s = []
        if type(t).__name__ == 'BinOp':
            if type(t.op).__name__ == 'Add':
                assert len(ops) == 0, 'Transformation of multiple terms is not supported in DTSR formula strings'
                s += self.process_ast(t.left, ops=ops, under_IRF=under_IRF, random_coefficients=random_coefficients)
                s += self.process_ast(t.right, ops=ops, under_IRF=under_IRF, random_coefficients=random_coefficients)
            elif type(t.op).__name__ == 'BitOr':
                assert len(ops) == 0, 'Transformation of random terms is not supported in DTSR formula strings'
                assert not under_IRF, 'Random terms may not be embedded under IRF terms in DTSR formula strings'
                assert random_coefficients is None, 'Random terms may not be embedded under other random terms in DTSR formula strings'
                random_coefficients = []
                s = self.process_ast(t.left, random_coefficients=random_coefficients)
                new = RandomTerm(s, t.right.id, random_coefficients[:])
                self.random[new.name()] = new
            elif type(t.op).__name__ == 'Mod':
                terms = self.process_ast(t.left, under_IRF=under_IRF, random_coefficients=random_coefficients) + self.process_ast(t.right, under_IRF=under_IRF, random_coefficients=random_coefficients)
                for x in terms:
                    if type(x).__name__ == 'IRFTreeNode':
                        raise ValueError('Interaction terms may not dominate IRF terms in DTSR formula strings')
                new = InteractionTerm(terms=terms, ops=ops)
                s.append(new)
            elif type(t.op).__name__ == 'Mult':
                assert len(ops) == 0, 'Transformation of term expansions is not supported in DTSR formula strings'
                left = self.process_ast(t.left, random_coefficients=random_coefficients)
                right = self.process_ast(t.right, random_coefficients=random_coefficients)
                terms = left + right
                for x in terms:
                    if type(x).__name__ == 'IRFTreeNode':
                        raise ValueError('Term expansions may not dominate IRF terms in DTSR formula strings')
                new = InteractionTerm(terms=terms, ops=ops)
                s += left + right + [new]
            elif type(t.op).__name__ == 'Pow':
                assert len(ops) == 0, 'Transformation of term expansions is not supported in DTSR formula strings'
                terms = self.process_ast(t.left, random_coefficients=random_coefficients)
                for x in terms:
                    if type(x).__name__ == 'IRFTreeNode':
                        raise ValueError('Term expansions may not dominate IRF terms in DTSR formula strings')
                order = min(int(t.right.n), len(terms))
                for i in range(1, order + 1):
                    collections = itertools.combinations(terms, i)
                    for tup in collections:
                        if i > 1:
                            new = InteractionTerm(list(tup), ops=ops)
                            s.append(new)
                        else:
                            s.append(tup[0])
        elif type(t).__name__ == 'Call':
            if t.func.id == 'C':
                assert len(t.args) == 2, 'C() takes exactly two arguments in DTSR formula strings'
                terms = self.process_ast(t.args[0], under_IRF=True, random_coefficients=random_coefficients)
                for x in terms:
                    new = self.process_irf(t.args[1], x, ops=None, under_IRF=under_IRF, random_coefficients=random_coefficients)
                    if new.name() not in self.preterminal_names:
                        self.preterminal_names.append(new.name())
                    if new.impulse.name() not in self.terminals:
                        self.terminals[new.impulse.name()] = new.impulse
                    s.append(new)
            elif t.func.id in Formula.IRF:
                raise ValueError('IRF calls can only occur as inputs to C() in DTSR formula strings')
            else:
                assert len(t.args) <= 1, 'Only unary ops on variables supported in DTSR formula strings'
                s += self.process_ast(t.args[0], ops=ops + [t.func.id], under_IRF=under_IRF, random_coefficients=random_coefficients)
        elif type(t).__name__ == 'Name':
            new = Term(t.id, ops=ops)
            s.append(new)
        elif type(t).__name__ == 'NameConstant':
            new = Term(t.value, ops=ops)
            s.append(new)
        elif type(t).__name__ == 'Num':
            new = Term(str(t.n), ops=ops)
            s.append(new)
        else:
            raise ValueError('Operation "%s" is not supported in DTSR formula strings' %type(t).__name___)
        return s

    def process_irf(self, t, impulse, ops=None, under_IRF=True, random_coefficients=None):
        if ops is None:
            ops = []
        assert t.func.id in Formula.IRF, 'Ill-formed model string: process_irf() called on non-IRF node'
        irf_id = None
        coef_id = None
        cont=False
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
        p = self.irf_tree
        if len(t.args) > 0:
            assert len(t.args) == 1, 'Ill-formed model string: IRF can take at most 1 positional argument'
            p = self.process_irf(t.args[0], impulse)
        new = IRFTreeNode(impulse=impulse, family=t.func.id, irfID=irf_id, coefID=coef_id, cont=cont, p=p, ops=ops)
        if new.family not in self.atomic_irf_by_family:
            self.atomic_irf_by_family[new.family] = []
        if new.irf_id() not in self.atomic_irf_by_family[new.family]:
            self.atomic_irf_by_family[new.family].append(new.irf_id())
        if not under_IRF and new.coef_id() not in self.coefficient_names:
            self.coefficient_names.append(new.coef_id())
        if random_coefficients is not None and new.coef_id() not in random_coefficients:
            random_coefficients.append(new.coef_id())
        if random_coefficients is None and not under_IRF and new.coef_id() not in self.fixed_coefficient_names:
            self.fixed_coefficient_names.append(new.coef_id())
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

    def apply_ops(self, term, df):
        ops = term.ops
        if term.name() not in df.columns:
            if term.id not in df.columns:
                if type(term).__name__ == 'InteractionTerm':
                    for t in term.terms:
                        df = self.apply_ops(t, df)
                    df[term.id] = df[[x.name() for x in term.terms]].product(axis=1)
                else:
                    raise ValueError('Unrecognized term "%s" in model formula' %term.id)
            else:
                df[term.name()] = df[term.id]
            for i in range(len(ops)):
                op = ops[i]
                df[term.name()] = self.apply_op(op, df[term.name()])
        return df

    def apply_formula(self, y, X):
        if self.dv not in y.columns:
            y = self.apply_ops(self.dv_term, y)
        for t in self.terminals:
            X = self.apply_ops(self.terminals[t], X)
        return y, X

    IRF = [
        'DiracDelta',
        'Exp',
        'ShiftedExp',
        'Gamma',
        'ShiftedGamma',
        'GammKgt1',
        'ShiftedGammaKgt1',
        'Normal',
        'SkewNormal',
        'EMG',
        'BetaPrime',
        'ShiftedBetaPrime',
    ]

    def __str__(self):
        return self.dv + ' ~ ' + ' + '.join([t.name() for t in self.terms])

class Term(object):
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

class InteractionTerm(object):
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

class RandomTerm(object):
    """
    A class representing a random effects term.
    
    # Arguments
        grouping_factor: String. A string representation of the random grouping factor
        vars: List of String. A list of fields for the random term
        intercept: Boolean. Whether to include an intercept for the random term
    """
    def __init__(self, terms, gf, coefficient_names):
        self.intercept = True
        self.terms = []
        while len(terms) > 0:
            if terms[0].name() == '0':
                self.intercept = False
            elif terms[0].name() == '1':
                self.intercept = True
            else:
                self.terms.append(terms[0])
            terms.pop(0)
        self.gf = gf
        self.name_str = '(' + ' + '.join([str(int(self.intercept))] + [t.name() for t in self.terms]) + ' | ' + self.gf + ')'
        self.id = self.name_str
        self.coefficient_names = coefficient_names[:]

    def name(self):
        return self.name_str

class IRFTreeNode(object):
    def __init__(self, impulse=None, family=None, irfID=None, coefID=None, ops=None, p=None, cont=False):
        if impulse is None:
            self.ops = []
            self.cont = False
            self.impulse = None
            self.family = None
            self.irfID = None
            self.coefID = None
            self.children = {}
            self.p = None
            self.fit_center = True
        else:
            if ops is None:
                ops = []
            self.ops = ops[:]
            self.cont = cont
            self.impulse = impulse
            self.family = family
            self.irfID = irfID
            self.coefID = coefID
            self.children = {}
            self.p = p
            self.fit_center = True

            if self.family not in p.children:
                p.children[self.family] = {}
            if self.name() not in p.children[self.family]:
                p.children[self.family][self.name()] = self
            else:
                self.merge_children(p.children[self.family][self.name()])
                p.children[self.family][self.name()] = self

    def name(self):
        if self.family is None:
            return 'ROOT'
        out = self.irf_str()
        out = 'C(' + self.impulse.name() + ', ' + out + ')'
        for op in self.ops:
            out = op + '(' + out + ')'
        return out

    def irf_str(self):
        if self.family is None:
            return ''
        out = self.family + '('
        args = []
        if self.p.name() != 'ROOT':
            args.append(self.p.irf_str())
        if self.irfID is not None:
            args.append('irf_id="%s"' % self.irfID)
        if self.coefID is not None:
            args.append('coef_id="%s"' % self.coefID)
        if self.cont:
            args.append('cont=%r' % self.cont)
        out += ', '.join(args)
        out += ')'
        return out

    def irf_id(self):
        if self.irfID is None:
            return self.name()
        return self.irfID

    def coef_id(self):
        if self.coefID is None:
            if self.terminal():
                return self.name()
            return None
        return self.coefID

    def terminal(self):
        return len(self.children) == 0

    def __str__(self):
        s = self.name()
        for f in self.children:
            s += '\n  %s'%f
            indent = '    '
            for c in self.children[f]:
                s += '\n%s' % indent + str(self.children[f][c]).replace('\n', '\n%s' % indent)
        if self.terminal():
            if self.impulse is not None:
                s += '\n  - %s' %self.impulse.name()
            else:
                return 'ROOT'
        return s

    def merge_children(self, new):
        for f in new.children:
            if f not in self.children:
                self.children[f] = {}
            for c in new.children[f]:
                if c not in self.children[f]:
                    self.children[f][c] = new.children[f][c]
