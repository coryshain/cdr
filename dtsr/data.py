import sys
import re
import numpy as np
import pandas as pd
from .util import names2ix

def build_DTSR_impulses(
        X,
        first_obs,
        last_obs,
        impulse_names,
        history_length=128,
        X_2d_predictor_names=None,
        X_2d_predictors=None,
        int_type='int32',
        float_type='float32',
):
    assert (X_2d_predictors is None and (X_2d_predictor_names is None or len(X_2d_predictor_names) == 0)) or (X_2d_predictors.shape[-1] == len(X_2d_predictor_names)), 'Shape mismatch between X_2d_predictors and X_2d_predictor_names'
    if X_2d_predictor_names is None:
        X_2d_predictor_names = []
    impulse_names_1d = list(set(impulse_names).difference(set(X_2d_predictor_names)))
    impulse_names_2d = list(set(impulse_names).intersection(set(X_2d_predictor_names)))
    assert len(impulse_names) == len(impulse_names_1d) + len(impulse_names_2d), 'Mismatch between 1d and 2d predictor sets'

    X_2d_from_1d, time_X_2d, time_mask = expand_history(
        X[impulse_names_1d],
        X.time,
        first_obs,
        last_obs,
        history_length,
        int_type=int_type,
        float_type=float_type
    )

    if X_2d_predictors is None:
        X_2d = X_2d_from_1d[:, :, names2ix(impulse_names, impulse_names_1d)]
    else:
        X_2d = np.concatenate([X_2d_from_1d, X_2d_predictors], axis=2)
        X_2d = X_2d[:, :, names2ix(impulse_names, impulse_names_1d + impulse_names_2d)]

    # ix = names2ix(impulse_names_1d, impulse_names)
    # print('Output check')
    # print(np.equal(X_2d[:,-1,ix], np.array(X[impulse_names_1d])[last_obs-1]).mean())
    # print(X_2d[:,-1,ix].shape)
    # print(X_2d[:10,-1,ix])
    # print(np.array(X[impulse_names_2d])[last_obs-1].shape)
    # print(np.array(X[impulse_names_2d])[last_obs-1][:10])

    return X_2d, time_X_2d, time_mask

def compute_history_intervals(X, y, series_ids):
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
                cor = ((x1 - x1_mean) * (x2 - x2_mean) * time_mask[..., 0]).sum() / \
                      np.sqrt(((x1 - x1_mean) ** 2 * time_mask[..., 0]).sum() * (
                              (x2 - x2_mean) ** 2 * time_mask[..., 0]).sum())
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
    time_mask = np.zeros((first_obs.shape[0], history_length, 1), dtype=FLOAT_NP)

    for i, first, last in zip(np.arange(first_obs.shape[0]), first_obs, last_obs):
        sX = X[first:last]
        sXt = X_time[first:last]
        X_2d[i, -sX.shape[0]:] = sX
        time_X_2d[i][-len(sXt):] = sXt
        time_mask[i][-len(sXt):, 0] = 1

    return X_2d, time_X_2d, time_mask

def compute_time_mask(X_time, first_obs, last_obs, history_length, int_type='int32', float_type='float32'):
    INT_NP = getattr(np, int_type)
    FLOAT_NP = getattr(np, float_type)
    last_obs = np.array(last_obs, dtype=INT_NP)
    first_obs = np.maximum(np.array(first_obs, dtype=INT_NP), last_obs - history_length)
    X_time = np.array(X_time, dtype=FLOAT_NP)

    time_mask = np.zeros((first_obs.shape[0], history_length, 1), dtype=FLOAT_NP)

    for i, first, last in zip(np.arange(first_obs.shape[0]), first_obs, last_obs):
        sXt = X_time[first:last]
        time_mask[i][-len(sXt):, 0] = 1

    return time_mask

def preprocess_data(X, y, p, formula_list, compute_history=True, debug=False):
    sys.stderr.write('Pre-processing data...\n')

    select = compute_filters(y, p.filter_map)

    y = y[select]

    X_2d_predictor_names = None
    X_2d_predictors = None

    if compute_history:
        sys.stderr.write('Computing history intervals for each regression target...\n')
        first_obs, last_obs = compute_history_intervals(X, y, p.series_ids)
        y['first_obs'] = first_obs
        y['last_obs'] = last_obs
        if debug:
            sample = np.random.randint(0, len(y), 10)
            for i in sample:
                print(i)
                row = y.iloc[i]
                print(row[['subject', 'docid', 'sentid', 'word']])
                print(X[['subject', 'docid', 'sentid', 'word']][row.first_obs:row.last_obs])

        for x in formula_list:
            X, y, X_2d_predictor_names, X_2d_predictors = x.apply_formula(
                X,
                y,
                X_2d_predictor_names=X_2d_predictor_names,
                X_2d_predictors=X_2d_predictors,
                history_length=p.history_length
            )

    return X, y, select, X_2d_predictor_names, X_2d_predictors

