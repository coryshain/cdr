import numpy as np
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

rstring = '''
    function(a, b) {
        return(anova(a, b))
    }
'''

anova = robjects.r(rstring)

rstring = '''
    function (x, mean = rep(0, p), sigma = diag(p), log = FALSE) {
        if (is.vector(x))
            x <- matrix(x, ncol = length(x))
        p <- ncol(x)
        if (!missing(mean)) {
            if (!is.null(dim(mean)))
                dim(mean) <- NULL
            if (length(mean) != p)
                stop("mean and sigma have non-conforming size")
        }
        if (!missing(sigma)) {
            if (p != ncol(sigma))
                stop("x and sigma have non-conforming size")
            if (!isSymmetric(sigma, tol = sqrt(.Machine$double.eps),
                check.attributes = FALSE))
                stop("sigma must be a symmetric matrix")
        }
        dec <- tryCatch(chol(sigma), error = function(e) e)
        if (inherits(dec, "error")) {
            x.is.mu <- colSums(t(x) != mean) == 0
            logretval <- rep.int(-Inf, nrow(x))
            logretval[x.is.mu] <- Inf
        }
        else {
            tmp <- backsolve(dec, t(x) - mean, transpose = TRUE)
            rss <- tmp^2
            logretval <- -log(diag(dec)) - 0.5 * log(2 *
                pi) - 0.5 * rss
        }
        names(logretval) <- rownames(x)
        if (log)
            logretval
        else exp(logretval)
    }
'''
dvector_mvnorm = robjects.r(rstring)
robjects.globalenv["dvector_mvnorm"] = dvector_mvnorm

class LM(object):
    """
    Ordinary least squares linear model

    :param: formula: ``str``; formula string defining the linear model.
    :param: X: ``pandas`` table or R dataframe; the training data.
    """
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
                return(predict(model, df, na.action=na.pass))
            }
        '''

        predict = robjects.r(rstring)

        return fit, summary, predict


class LME(object):
    NO_CONVERGENCE_WARNINGS = 'No convergence warnings.'

    def __init__(self, formula, X):
        lmer = importr('lme4')
        fit, summary, convergence_warnings, predict, log_lik_base = self.instance_methods()
        self.m = fit(formula, X)
        self.convergence_warnings = lambda: '\n'.join([str(x) for x in convergence_warnings(self.m)])
        self.summary = lambda: str(summary(self.m)) + '\nConvergence warnings:\n%s\n\n' % self.convergence_warnings()
        self.predict = lambda x: predict(self.m, x)
        self.log_lik = lambda newdata=None, summed=True: log_lik_base(self.m, newdata=newdata, summed=summed)

    def __getstate__(self):
        return self.m

    def converged(self):
        warnings = self.convergence_warnings()
        return warnings == LME.NO_CONVERGENCE_WARNINGS

    def __setstate__(self, state):
        lmer = importr('lme4')
        self.m = state
        fit, summary, convergence_warnings, predict, log_lik_base = self.instance_methods()
        self.convergence_warnings = lambda: '\n'.join([str(x) for x in convergence_warnings(self.m)])
        self.summary = lambda: str(summary(self.m)) + '\nConvergence warnings:\n%s\n\n' % self.convergence_warnings()
        self.predict = lambda x: predict(self.m, x)
        self.log_lik = lambda newdata=None, summed=True: np.array(log_lik_base(self.m, newdata=newdata, summed=summed))

    def instance_methods(self):
        rstring = '''
            function(bform, df) {
                print('LME model!!!!')
                print(head(df))
                return(lmer(bform, data=df, REML=FALSE))
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
            function(model) {
                convWarn <- model@optinfo$conv$lme4$messages
                if (is.null(convWarn)) {
                    convWarn <- "%s"
                }
                return(convWarn)
            }
        ''' % LME.NO_CONVERGENCE_WARNINGS

        convergence_warnings = robjects.r(rstring)

        rstring = '''
            function(model, df) {
                return(predict(model, df, allow.new.levels=TRUE, na.action=na.pass))
            }
        '''

        predict = robjects.r(rstring)

        rstring = '''   
            logLik2 <- function(model, newdata=NULL, summed=TRUE) {
                if (is.null(newdata) && summed) {
                    return(logLik(m))
                } else {
                    library(mvtnorm)   
                    if (is.null(newdata)) {
                        newdata = m@frame
                    }
                    
                    dv <- as.character(formula(model))[[2]]
                    if (! dv %in% colnames(newdata)) {
                        dv <- gsub('(','.', gsub(')', '.', dv, fixed=TRUE), fixed=TRUE)
                    }
                    dv <- newdata[[dv]]
                                
                    z <- getME(model, "Z")
                    zt <- getME(model, "Zt")
                    psi <- list()
                    for (i in 1:length(names(model@flist))) {
                        psi[[i]] <- replicate(length(unique(newdata[[names(model@flist)[[i]]]])), VarCorr(model)[[names(model@flist)[[i]]]], simplify = FALSE)
                    }
                    psi <- Reduce(c, psi)
                    psi = bdiag(psi)
            
                    betw <- z %*% psi %*% zt
                    err <- Diagonal(nrow(newdata), sigma(model) ^ 2)
                    v <- betw + err
            
                    preds <- predict(model, newdata=newdata, allow.new.levels=TRUE, re.form = NA)
            
                    lls <- dvector_mvnorm(dv, preds, as.matrix(v), log = TRUE)
                    if (summed) {
                        return(sum(lls))
                    } else {
                        return(lls)
                    }
                }
            }
        '''
        log_lik = robjects.r(rstring)

        return fit, summary, convergence_warnings, predict, log_lik

class GAM(object):
    def __init__(self, formula, X, ran_gf=None):
        mgcv = importr('mgcv')
        process_ran_gf, add_z, add_log, fit, summary, predict, unique = self.instance_methods()
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
        if ran_gf is not None:
            X = process_ran_gf(X, ran_gf)
        X = add_z(X)
        X = add_log(X)
        self.m = fit(formula, X)
        self.summary = lambda: summary(self.m)
        self.predict = lambda x: predict(self.m, self.formula, x, self.subject, self.word)

    def __getstate__(self):
        return (self.m, self.subject, self.word, self.formula)

    def __setstate__(self, state):
        mgcv = importr('mgcv')
        self.m, self.subject, self.word, self.formula = state
        process_ran_gf, add_z, add_log, fit, summary, predict, unique = self.instance_methods()
        self.summary = lambda: summary(self.m)
        self.predict = lambda x: predict(self.m, self.formula, x, self.subject, self.word)

    def instance_methods(self):
        rstring = '''
            function(X, ran_gf) {
                for (gf in ran_gf) {
                    X[[gf]] <- as.factor(X[[gf]])
                }
                return(X)
            }
        '''
        process_ran_gf = robjects.r(rstring)

        rstring = '''
            function(X) {
                for (c in names(X)) {
                    if (is.numeric(X[[c]])) {
                        X[[paste0('z_',c)]] <- scale(X[[c]])
                    }
                }
                return(X)
            }
        '''
        add_z = robjects.r(rstring)

        rstring = '''
            function(X) {
                for (c in names(X)) {
                    if (is.numeric(X[[c]])) {
                        X[[paste0('log_',c)]] <- log(X[[c]])
                    }
                }
                return(X)
            }
        '''
        add_log = robjects.r(rstring)

        rstring = '''
            function(bform, df) {
                return(bam(as.formula(bform), data=df, drop.unused.levels=FALSE, nthreads=10))
            }
        '''

        fit = robjects.r(rstring)

        rstring = '''
            function(model) {
                return(summary(model))
            }
        '''

        summary = robjects.r(rstring)

        rstring = '''
            function(model, bform, df, subjects=NULL, words=NULL) {
                for (c in names(df)) {
                    if (is.numeric(df[[c]])) {
                        df[[paste0('z_',c)]] <- scale(df[[c]])
                    }
                }
                for (c in names(df)) {
                    if (is.numeric(df[[c]])) {
                        df[[paste0('log_',c)]] <- scale(df[[c]])
                    }
                }
                select = logical(nrow(df))
                select = !select
                if (grepl('subject', bform) & !is.null(subjects)) {
                    select = select & df$subject %in% subjects
                }
                grepl('word', bform)
                if (grepl('word', bform) & !is.null(words)) {
                    select = select & (word %in% words)
                }
                preds = predict(model, df[select,], na.action=na.pass)
                df$preds = NA
                df[select,]$preds = preds
                return(df$preds)
            }
        '''

        predict = robjects.r(rstring)

        rstring = '''
            function(df, col) {
                return(unique(df[[col]]))
            }
        '''

        unique = robjects.r(rstring)

        return process_ran_gf, add_z, add_log, fit, summary, predict, unique

def py2ri(x):
    return pandas2ri.py2ri(x)
