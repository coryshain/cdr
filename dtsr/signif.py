import numpy as np
import math

def bootstrap(true, preds_1, preds_2, err_type='mse', n_iter=10000, n_tails=2):
    err_table = np.stack([preds_1, preds_2], 1)
    err_table -= np.expand_dims(true, -1)
    if err_type.lower == 'mae':
        err_table = err_table.abs()
    else:
        err_table *= err_table
    base_diff = err_table[:,0].mean() - err_table[:,1].mean()
    hits = 0
    for i in range(n_iter):
        shuffle = (np.random.random((len(true))) > 0.5).astype('int')
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
    return p