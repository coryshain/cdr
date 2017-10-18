import sys
import numpy as np

def compute_history_intervals(y, X, series_ids, cutoff=100):
    id_vectors_X = np.zeros((len(X), len(series_ids))).astype('int32')
    id_vectors_y = np.zeros((len(y), len(series_ids))).astype('int32')

    n = len(y)

    time_X = np.array(X.time)
    time_y = np.array(y.time)

    for i in range(len(series_ids)):
        col = series_ids[i]
        id_vectors_X[:, i] = np.array(X[col].cat.codes)
        id_vectors_y[:, i] = np.array(y[col].cat.codes)
    cur_ids = id_vectors_y[0]

    first_obs = np.zeros(len(y)).astype('int32')
    last_obs = np.zeros(len(y)).astype('int32')

    i = j = 0
    start = 0
    end = 0
    while i < n and j < len(X):
        sys.stderr.write('\r%d/%d' %(i+1, n))
        sys.stderr.flush()
        if (id_vectors_y[i] != cur_ids).any():
            start = end = j
            cur_ids = id_vectors_y[i]
        while j < len(X) and not (id_vectors_X[j] == cur_ids).all():
            start += 1
            end += 1
            j += 1
        while j < len(X) and time_X[j] <= time_y[i] and (id_vectors_X[j] == cur_ids).all():
            end += 1
            j += 1
        first_obs[i] = start
        last_obs[i] = end

        i += 1

    sys.stderr.write('\n')
    return first_obs, last_obs

def compute_filters(y, filter_map=None):
    if filter_map is None:
        return y
    select = np.ones(len(y), dtype=bool)
    for field in filter_map:
        for cond in filter_map[field]:
            select &= compute_filter(y, field, cond)
    return select

def compute_filter(y, field, cond):
    cond = cond.strip()
    if cond.startswith('<='):
        return y[field] <= float(cond[2:].strip())
    if cond.startswith('>='):
        return y[field] >= float(cond[2:].strip())
    if cond.startswith('<'):
        return y[field] < float(cond[1:].strip())
    if cond.startswith('>'):
        return y[field] > float(cond[1:].strip())
    if cond.startswith('=='):
        try:
            return y[field] == float(cond[2:].strip())
        except:
            return y[field].astype('str') == cond[2:].strip()
    if cond.startswith('!='):
        try:
            return y[field] != float(cond[2:].strip())
        except:
            return y[field].astype('str') != cond[2:].strip()
    raise ValueError('Unsupported comparator in filter "%s"' %cond)

def compute_splitID(y, split_fields):
    splitID = np.zeros(len(y), dtype='int32')
    for col in split_fields:
        splitID += y[col].cat.codes
    return splitID

def compute_partition(y, modulus, n):
    partition = [((y.splitID) % modulus) <= (modulus - n)]
    for i in range(n-1, 0, -1):
        partition.append(((y.splitID) % modulus) == (modulus - i))
    return partition

def preprocess_data(X, y, p, formula_list, compute_history=True):
    sys.stderr.write('Pre-processing data...\n')

    for x in formula_list:
        if len(x.random) > 0:
            n_level = []
            for r in x.random:
                gf = r.grouping_factor
                n_level.append(len(y[gf].cat.categories))

    select = compute_filters(y, p.filter_map)

    y = y[select]
    for x in formula_list:
        y, X = x.apply_formula(y, X)

    if compute_history:
        sys.stderr.write('Computing history intervals for each regression target...\n')
        first_obs, last_obs = compute_history_intervals(y, X, p.series_ids)
        y['first_obs'] = first_obs
        y['last_obs'] = last_obs
        if False:
            sample = np.random.randint(0, len(y), 10)
            for i in sample:
                print(i)
                row = y.iloc[i]
                print(row[['subject', 'docid', 'sentid', 'word']])
                print(X[['subject', 'docid', 'sentid', 'word']][row.first_obs:row.last_obs])
    return X, y, select

