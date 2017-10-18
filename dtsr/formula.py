import sys
import numpy as np

def z(df):
    return (df-df.mean(axis=0))/df.std(axis=0)

def c(df):
    return df-df.mean(axis=0)

class Formula(object):
    """
    A class for parsing R-style mixed-effects model formula strings and applying them to DTSR design matrices.
    
    # Arguments
        bform_str: String. An R-style mixed-effects model formula string
    """
    def __init__(self, bform_str):
        n_lp = bform_str.count('(')
        n_rp = bform_str.count(')')
        assert n_lp == n_rp, 'Unmatched parens in formula "%s".' % bform_str

        self.dv = ''
        self.fixed = ['']
        self.random = []
        self.dv, bform_str = bform_str.strip().split('~')
        self.dv = self.dv.strip()

        in_random_term = False
        in_grouping_factor = False
        for c in bform_str.strip():
            if c == '(' and self.fixed[-1] == '':
                in_random_term = True
                if self.fixed[-1] == '':
                    self.fixed.pop(-1)
                self.random.append(RandomTerm())
            elif c == ')' and in_random_term:
                self.n_lb = self.random[-1].vars[-1].count('(')
                self.n_rb = self.random[-1].vars[-1].count(')')
                if self.n_lb == self.n_rb:
                    in_random_term = False
                    in_grouping_factor = False
                else:
                    self.random[-1].vars[-1] += c
            elif c in [' ', '+', '|']:
                if c == '+':
                    if in_random_term:
                        self.random[-1].vars.append('')
                    else:
                        self.fixed.append('')
                elif c == '|':
                    in_grouping_factor = True
            else:
                if in_random_term:
                    if in_grouping_factor:
                        self.random[-1].grouping_factor += c
                    else:
                        self.random[-1].vars[-1] += c
                else:
                    self.fixed[-1] += c
        for i in range(len(self.random)):
            if '0' in self.random[i].vars:
                self.random[i].intercept = False
            self.random[i].vars = [v for v in self.random[i].vars if v not in ['0', '1']]

        self.ransl = []
        for f in self.random:
            for v in f.vars:
                if f not in self.ransl:
                    self.ransl.append(v)
        self.rangf = []
        for f in self.random:
            if f.grouping_factor not in self.rangf:
                self.rangf.append(f.grouping_factor)
        self.allsl = self.fixed[:]
        for f in self.ransl:
            if f not in self.allsl:
                self.allsl.append(f)
        self.allvar = self.allsl[:]
        for gf in self.rangf:
            if gf not in self.allvar:
                self.allvar.append(gf)

    def parse_var(self, var):
        n_lp = var.count('(')
        n_rp = var.count(')')
        assert n_lp == n_rp, 'Unmatched parens in formula variable "%s".' % var
        ops = [[], var.strip()]
        while ops[1].endswith(')'):
            op, inp = self.parse_var_inner(ops[1])
            ops[0].insert(0, op)
            ops[1] = inp.strip()
        return ops

    def parse_var_inner(self, var):
        if var.strip().endswith(')'):
            op = ''
            inp = ''
            inside_var = False
            for i in var[:-1]:
                if inside_var:
                    inp += i
                else:
                    if i == '(':
                        inside_var = True
                    else:
                        op += i
        else:
            op = ''
            inp = var
        return op, inp

    def extract_cross(self, var):
        vars = var.split(':')
        if len(vars) > 1:
            vars = [v.strip() for v in vars]
        return vars

    def extract_interaction(var):
        vars = var.split('*')
        if len(vars) > 1:
            vars = [v.strip() for v in vars]
        return vars

    def process_interactions(self, vars):
        new_vars = []
        interactions = []
        for v1 in vars:
            new_v = self.extract_cross(v1)
            if len(new_v) > 0:
                interactions.append(new_v)
            new_v = self.extract_interaction(v1)
            if len(new_v) > 0:
                interactions.append(new_v)
                for v2 in new_v:
                    if v2 not in new_vars:
                        new_vars.append(v2)
        return new_vars, interactions

    def apply_op(self, op, arr):
        if op in ['c', 'c.']:
            out = c(arr)
        elif op in ['z', 'z.']:
            out = z(arr)
        elif op == 'log':
            out = np.log(arr)
        elif op == 'log1p':
            out = np.log(arr + 1)
        elif op.startswith('pow'):
            exponent = float(op[3:])
            out = arr ** exponent
        elif op.startswith('bc'):
            L = float(op[2:])
            if L == 0:
                out = np.log(arr)
            else:
                out = (arr ** L - 1) / L
        else:
            raise ValueError('Unrecognized op: "%s".' % op)
        return out

    def apply_ops(self, ops, col, df):
        if len(ops[0]) > 0:
            new_col = ops[-1]
            for op in ops[0]:
                new_col = op + '(' + new_col + ')'
            df[new_col] = df[col]
            for op in ops[0]:
                df[new_col] = self.apply_op(op, df[new_col])
        return df

    def apply_ops_from_str(self, s, df):
        ops = self.parse_var(s)
        col = ops[1]
        df = self.apply_ops(ops, col, df)
        return df

    def apply_formula(self, y, X):
        if self.dv not in y.columns:
            y = self.apply_ops_from_str(self.dv, y)
        for f in self.fixed:
            if f not in X.columns:
                X = self.apply_ops_from_str(f, X)
        for r in self.random:
            for v in r.vars:
                if f not in X.columns:
                    X = self.apply_ops_from_str(v, X)
        return y, X

    def variables(self):
        dv = self.dv
        fixef = self.fixed[:]
        ransl = []
        for f in self['random']:
            for v in f['vars']:
                if f not in ransl:
                    ransl.append(v)
        rangf = []
        for f in self['random']:
            if f['grouping_factor'] not in rangf:
                rangf.append(f['grouping_factor'])
        allsl = fixef[:]
        for f in ransl:
            if f not in allsl:
                allsl.append(f)
        allvar = allsl[:]
        for gf in rangf:
            if gf not in allvar:
                allvar.append(gf)

        return ransl, rangf, allsl, allvar

class RandomTerm(object):
    """
    A class representing a random effects term.
    
    # Arguments
        grouping_factor: String. A string representation of the random grouping factor
        vars: List of String. A list of fields for the random term
        intercept: Boolean. Whether to include an intercept for the random term
    """
    def __init__(self, grouping_factor='', vars=None, intercept=True):
        self.grouping_factor = grouping_factor
        if vars is None:
            self.vars = ['']
        else:
            self.vars = vars
        self.intercept = intercept