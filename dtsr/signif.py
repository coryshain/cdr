import sys
import numpy as np
import math

def bootstrap(err_1, err_2, n_iter=10000, n_tails=2):
    err_table = np.stack([err_1, err_2], 1)
    base_diff = err_table[:,0].mean() - err_table[:,1].mean()
    hits = 0
    sys.stderr.write('Permutation testing...\n')
    for i in range(n_iter):
        sys.stderr.write('\r%d/%d' %(i+1, n_iter))
        sys.stderr.flush()
        shuffle = (np.random.random((len(err_table))) > 0.5).astype('int')
        m1 = err_table[np.arange(len(err_table)),shuffle]
        m2 = err_table[np.arange(len(err_table)),1-shuffle]
        cur_diff = m1.mean() - m2.mean()
        if n_tails == 1:
            if base_diff < 0 and cur_diff <= base_diff:
                hits += 1
            elif base_diff > 0 and cur_diff >= base_diff:
                hits += 1
            elif base_diff == 0:
                hits += 1
        elif n_tails == 2:
            if math.fabs(cur_diff) > math.fabs(base_diff):
                hits += 1
        else:
            raise ValueError('Invalid bootstrap parameter n_tails: %s. Must be in {0, 1}.' %n_tails)

    p = float(hits+1)/(n_iter+1)

    sys.stderr.write('\n')

    return p, base_diff