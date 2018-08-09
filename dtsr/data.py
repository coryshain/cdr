import sys
import numpy as np
import pandas as pd
from .util import names2ix

def z(df):
    return (df-df.mean(axis=0))/df.std(axis=0)

def c(df):
    return df-df.mean(axis=0)

def s(df):
    return df/df.std(axis=0)

def filter_invalid_responses(y, dv):
    select_y_valid = np.isfinite(y[dv]) & (y.last_obs > y.first_obs)
    return y[select_y_valid], select_y_valid

def build_DTSR_impulses(
        X,
        first_obs,
        last_obs,
        impulse_names,
        history_length=128,
        X_response_aligned_predictor_names=None,
        X_response_aligned_predictors=None,
        X_2d_predictor_names=None,
        X_2d_predictors=None,
        int_type='int32',
        float_type='float32',
):
    if X_response_aligned_predictor_names is None:
        X_response_aligned_predictor_names = []
    assert (X_2d_predictors is None and (X_2d_predictor_names is None or len(X_2d_predictor_names) == 0)) or (X_2d_predictors.shape[-1] == len(X_2d_predictor_names)), 'Shape mismatch between X_2d_predictors and X_2d_predictor_names'
    if X_2d_predictor_names is None:
        X_2d_predictor_names = []

    impulse_names_1d = sorted(list(set(impulse_names).difference(set(X_response_aligned_predictor_names)).difference(set(X_2d_predictor_names))))

    X_2d_from_1d, time_X_2d, time_mask = expand_history(
        X[impulse_names_1d],
        X.time,
        first_obs,
        last_obs,
        history_length,
        int_type=int_type,
        float_type=float_type
    )

    X_2d = X_2d_from_1d

    if X_response_aligned_predictors is not None:
        print(X_response_aligned_predictors.columns)
        X_response_aligned_predictors_new = np.zeros((X_2d_from_1d.shape[0], X_2d_from_1d.shape[1], len(X_response_aligned_predictor_names)))
        X_response_aligned_predictors_new[:, -1, :] = X_response_aligned_predictors[X_response_aligned_predictor_names]
        X_2d = np.concatenate([X_2d, X_response_aligned_predictors_new], axis=2)

    if X_2d_predictors is not None:
        X_2d = np.concatenate([X_2d, X_2d_predictors], axis=2)

    X_2d = X_2d[:,:,names2ix(impulse_names, impulse_names_1d + X_response_aligned_predictor_names + X_2d_predictor_names)]

    return X_2d, time_X_2d, time_mask

def compute_history_intervals(X, y, series_ids):
    m = len(X)
    n = len(y)

    time_X = np.array(X.time)
    time_y = np.array(y.time)

    id_vectors_X = []
    id_vectors_y = []

    for i in range(len(series_ids)):
        col = series_ids[i]
        id_vectors_X.append(np.array(X[col]))
        id_vectors_y.append(np.array(y[col]))
    id_vectors_X = np.stack(id_vectors_X, axis=1)
    id_vectors_y = np.stack(id_vectors_y, axis=1)

    y_cur_ids = id_vectors_y[0]

    first_obs = np.zeros(len(y)).astype('int32')
    last_obs = np.zeros(len(y)).astype('int32')

    # i iterates y
    i = 0
    # j iterates X
    j = 0
    start = 0
    end = 0
    epsilon = np.finfo(np.float32).eps
    while i < n:
        sys.stderr.write('\r%d/%d' %(i+1, n))
        sys.stderr.flush()

        # Check if we've entered a new series in y
        if (id_vectors_y[i] != y_cur_ids).any():
            start = end = j
            X_cur_ids = id_vectors_X[j]
            y_cur_ids = id_vectors_y[i]

        # Move the X pointer forward until we are either in the same series as y or at the end of the table.
        # However, if we are already at the end of the current time series, stay put in case there are subsequent observations of the response.
        if j == 0 or (j > 0 and (id_vectors_X[j-1] != y_cur_ids).any()):
            while j < m and (id_vectors_X[j] != y_cur_ids).any():
                j += 1
                start = end = j

        # Move the X pointer forward until we are either at the end of the series or have moved later in time than y
        while j < m and time_X[j] <= (time_y[i] + epsilon) and (id_vectors_X[j] == y_cur_ids).all():
            j += 1
            end = j

        first_obs[i] = start
        last_obs[i] = end

        i += 1

    sys.stderr.write('\n')
    
    return first_obs, last_obs

def corr_dtsr(X_2d, impulse_names, impulse_names_2d, time_mask):
    rho = pd.DataFrame(np.zeros((len(impulse_names), len(impulse_names))), index=impulse_names, columns=impulse_names)

    n_2d = X_2d.shape[0]
    n_3d = time_mask.sum()

    for i in range(len(impulse_names)):
        for j in range(i, len(impulse_names)):
            if impulse_names[i] in impulse_names_2d or impulse_names[j] in impulse_names_2d:
                x1 = X_2d[..., i]
                x2 = X_2d[..., j]

                n = n_3d
                x1_mean = x1.sum() / n
                x2_mean = x2.sum() / n
                cor = ((x1 - x1_mean) * (x2 - x2_mean) * time_mask).sum() / \
                      np.sqrt(((x1 - x1_mean) ** 2 * time_mask).sum() * (
                              (x2 - x2_mean) ** 2 * time_mask).sum())
            else:
                x1 = X_2d[:, -1, i]
                x2 = X_2d[:, -1, j]

                n = n_2d
                x1_mean = x1.sum() / n
                x2_mean = x2.sum() / n
                cor = ((x1 - x1_mean) * (x2 - x2_mean)).sum() / \
                      np.sqrt(((x1 - x1_mean) ** 2).sum() * ((x2 - x2_mean) ** 2).sum())

            rho.loc[impulse_names[i], impulse_names[j]] = cor
            if i != j:
                rho.loc[impulse_names[j], impulse_names[i]] = cor

    return rho

def compute_filters(y, filter_map=None):
    if filter_map is None:
        return y
    select = np.ones(len(y), dtype=bool)
    for field in filter_map:
        if field in y.columns:
            for cond in filter_map[field]:
                select &= compute_filter(y, field, cond)
    return select

def compute_filter(y, field, cond):
    cond = cond.strip()
    if cond.startswith('<='):
        return y[field] <= (np.inf if cond[2:].strip() == 'inf' else float(cond[2:].strip()))
    if cond.startswith('>='):
        return y[field] >= (np.inf if cond[2:].strip() == 'inf' else float(cond[2:].strip()))
    if cond.startswith('<'):
        return y[field] < (np.inf if cond[1:].strip() == 'inf' else float(cond[1:].strip()))
    if cond.startswith('>'):
        return y[field] > (np.inf if cond[1:].strip() == 'inf' else float(cond[1:].strip()))
    if cond.startswith('=='):
        try:
            return y[field] == (np.inf if cond[2:].strip() == 'inf' else float(cond[2:].strip()))
        except:
            return y[field].astype('str') == cond[2:].strip()
    if cond.startswith('!='):
        try:
            return y[field] != (np.inf if cond[2:].strip() == 'inf' else float(cond[2:].strip()))
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

def expand_history(X, X_time, first_obs, last_obs, history_length, int_type='int32', float_type='float32', fill=0.):
    INT_NP = getattr(np, int_type)
    FLOAT_NP = getattr(np, float_type)
    last_obs = np.array(last_obs, dtype=INT_NP)
    first_obs = np.maximum(np.array(first_obs, dtype=INT_NP), last_obs - history_length)
    X_time = np.array(X_time, dtype=FLOAT_NP)
    X = np.array(X)

    X_2d = np.full((first_obs.shape[0], history_length, X.shape[1]), fill)
    time_X_2d = np.zeros((first_obs.shape[0], history_length), dtype=FLOAT_NP)
    time_mask = np.zeros((first_obs.shape[0], history_length), dtype=FLOAT_NP)

    for i, first, last in zip(np.arange(first_obs.shape[0]), first_obs, last_obs):
        sX = X[first:last]
        sXt = X_time[first:last]
        X_2d[i, -sX.shape[0]:] = sX
        time_X_2d[i][-len(sXt):] = sXt
        time_mask[i][-len(sXt):] = 1

    return X_2d, time_X_2d, time_mask

def compute_time_mask(X_time, first_obs, last_obs, history_length, int_type='int32', float_type='float32'):
    INT_NP = getattr(np, int_type)
    FLOAT_NP = getattr(np, float_type)
    last_obs = np.array(last_obs, dtype=INT_NP)
    first_obs = np.maximum(np.array(first_obs, dtype=INT_NP), last_obs - history_length)
    X_time = np.array(X_time, dtype=FLOAT_NP)

    time_mask = np.zeros((first_obs.shape[0], history_length), dtype=FLOAT_NP)

    for i, first, last in zip(np.arange(first_obs.shape[0]), first_obs, last_obs):
        sXt = X_time[first:last]
        time_mask[i][-len(sXt):] = 1

    return time_mask

def preprocess_data(X, y, p, formula_list, compute_history=True, debug=False):
    sys.stderr.write('Pre-processing data...\n')

    if hasattr(p, 'filter_map'):
        select = compute_filters(y, p.filter_map)
        y = y[select]
    else:
        select = np.full((len(y),), True, dtype='bool')

    X_response_aligned_predictor_names = None
    X_response_aligned_predictors = None
    X_2d_predictor_names = None
    X_2d_predictors = None

    if compute_history:
        sys.stderr.write('Computing history intervals for each regression target...\n')
        first_obs, last_obs = compute_history_intervals(X, y, p.series_ids)
        y['first_obs'] = first_obs
        y['last_obs'] = last_obs

        # Floating point precision issues can allow the response to precede the impulse for simultaneous X/y,
        # which can break downstream convolution. The correction below to y.time prevents this.
        y.time = np.where(last_obs > first_obs, np.maximum(np.array(X.time)[last_obs - 1], y.time), y.time)

        if debug:
            sample = np.random.randint(0, len(y), 10)
            sample = np.concatenate([np.zeros((1,), dtype='int'), sample, np.ones((1,), dtype='int') * (len(y)-1)], axis=0)
            for i in sample:
                print(i)
                row = y.iloc[i]
                print(row[['subject', 'docid', 'time']])
                print(X[['subject', 'docid', 'word', 'time']][row.first_obs:row.last_obs])

        for x in formula_list:
            X, y, X_response_aligned_predictor_names, X_response_aligned_predictors, X_2d_predictor_names, X_2d_predictors = x.apply_formula(
                X,
                y,
                X_2d_predictor_names=X_2d_predictor_names,
                X_2d_predictors=X_2d_predictors,
                X_response_aligned_predictor_names=X_response_aligned_predictor_names,
                X_response_aligned_predictors=X_response_aligned_predictors,
                history_length=p.history_length
            )

    return X, y, select, X_response_aligned_predictor_names, X_response_aligned_predictors, X_2d_predictor_names, X_2d_predictors

