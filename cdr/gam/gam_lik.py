import sys
import pickle
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

rstring = '''function(model) {return(loglik(model))}'''
loglik_fn = robjects.r(rstring)

for path in sys.argv[1:]:
    with open(path, 'rb') as f:
        m = pickle.load(f)
        ll = loglik_fn(m)
        print('%s: %s' % (path, ll))

