import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

rstring = '''
     function(x) {
         return(scale(x, scale=FALSE))
     }
'''

center = robjects.r(rstring)
robjects.globalenv["c."] = center

rstring = '''
     function(x) {
         return(scale(x, scale=TRUE))
     }
'''

z_score = robjects.r(rstring)
robjects.globalenv["z."] = z_score

rstring = '''
     function(x) {
         return(x/sd(x))
     }
'''

scale = robjects.r(rstring)
robjects.globalenv["s."] = scale

class LM(object):
    def __init__(self, formula, X):
        fit, summary, predict = self.instance_methods()
        self.m = fit(formula, X)
        self.summary = lambda: summary(self.m)
        self.predict = lambda x: predict(self.m, x)

    def __getstate__(self):
        return self.m

    def __setstate__(self, state):
        self.m = state
        fit, summary, predict = self.instance_methods()
        self.summary = lambda: summary(self.m)
        self.predict = lambda x: predict(self.m, x)

    def instance_methods(self):
        rstring = '''
            function(bform, df) {
                return(lm(bform, data=df, REML=FALSE))
            }
        '''

        fit = robjects.r(rstring)

        rstring = '''
            function(model) {
                s = summary(model)
                return(s)
            }
        '''

        summary = robjects.r(rstring)

        rstring = '''
            function(model, df) {
                return(predict(model, df, allow.new.levels=TRUE))
            }
        '''

        predict = robjects.r(rstring)

        return fit, summary, predict


class LME(object):
    def __init__(self, formula, X):
        lmer = importr('lme4')
        fit, summary, predict = self.instance_methods()
        self.m = fit(formula, X)
        self.summary = lambda: summary(self.m)
        self.predict = lambda x: predict(self.m, x)

    def __getstate__(self):
        return self.m

    def __setstate__(self, state):
        lmer = importr('lme4')
        self.m = state
        fit, summary, predict = self.instance_methods()
        self.summary = lambda: summary(self.m)
        self.predict = lambda x: predict(self.m, x)

    def instance_methods(self):
        rstring = '''
            function(bform, df) {
                return(lmer(bform, data=df, REML=FALSE))
            }
        '''

        fit = robjects.r(rstring)

        rstring = '''
            function(model) {
                s = summary(model)
                convWarn <- model@optinfo$conv$lme4$messages
                if (is.null(convWarn)) {
                    convWarn <- 'No convergence warnings.'
                }
                s$convWarn = convWarn
                return(s)
            }
        '''

        summary = robjects.r(rstring)

        rstring = '''
            function(model, df) {
                return(predict(model, df, allow.new.levels=TRUE))
            }
        '''

        predict = robjects.r(rstring)

        return fit, summary, predict

class GAM(object):
    def __init__(self, formula, X):
        mgcv = importr('mgcv')
        fit, summary, predict, unique = self.instance_methods()
        self.formula = formula
        rstring = '''
            function() numeric()
        '''
        empty = robjects.r(rstring)
        if 'subject' in self.formula:
            self.subject = unique(X, 'subject')
        else:
            self.subject = empty
        if 'word' in self.formula:
            self.word = unique(X, 'word')
        else:
            self.word = empty()
        self.m = fit(formula, X)
        self.summary = lambda: summary(self.m)
        self.predict = lambda x: predict(self.m, self.formula, x, self.subject, self.word)

    def __getstate__(self):
        return (self.m, self.subject, self.word, self.formula)

    def __setstate__(self, state):
        mgcv = importr('mgcv')
        self.m, self.subject, self.word, self.formula = state
        fit, summary, predict, unique = self.instance_methods()
        self.summary = lambda: summary(self.m)
        self.predict = lambda x: predict(self.m, self.formula, x, self.subject, self.word)

    def instance_methods(self):
        rstring = '''
            function(bform, df) {
                return(gam(as.formula(bform), data=df, drop.unused.levels=FALSE))
            }
        '''

        fit = robjects.r(rstring)

        rstring = '''
            function(model) {10000
                return(summary(model))
            }
        '''

        summary = robjects.r(rstring)

        rstring = '''
            function(model, bform, df, subjects=NULL, words=NULL) {
                select = logical(nrow(df))
                select = !select
                if (grepl('subject', bform) & !is.null(subjects)) {
                    select = select & df$subject %in% subjects
                }
                grepl('word', bform)
                if (grepl('word', bform) & !is.null(words)) {
                    select = select & (word %in% words)
                }
                preds = predict(model, df[select,])
                df$preds = NA
                df[select,]$preds = preds
                return(df$preds)
            }
        '''

        rstring = '''
            function(model, bform, df, subjects=NULL, words=NULL) {
                preds = predict(model, df[select,])
                return(preds)
            }
        '''

        predict = robjects.r(rstring)

        rstring = '''
            function(df, col) {
                return(unique(df[[col]]))
            }
        '''

        unique = robjects.r(rstring)

        return fit, summary, predict, unique

def py2ri(x):
    return pandas2ri.py2ri(x)