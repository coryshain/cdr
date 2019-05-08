import sys
import re
import numpy as np
import pandas as pd
from .util import names2ix

op_finder = re.compile('([^()]+)\((.+)\) *')


def z(df):
    """
    Z-transform pandas series or data frame

    :param df: ``pandas`` ``Series`` or ``DataFrame``; input date
    :return: ``pandas`` ``Series`` or ``DataFrame``; z-transformed data
    """

    return (df-df.mean(axis=0))/df.std(axis=0)


def c(df):
    """
    Zero-center pandas series or data frame

    :param df: ``pandas`` ``Series`` or ``DataFrame``; input date
    :return: ``pandas`` ``Series`` or ``DataFrame``; centered data
    """

    return df-df.mean(axis=0)


def s(df):
    """
    Rescale pandas series or data frame by its standard deviation

    :param df: ``pandas`` ``Series`` or ``DataFrame``; input date
    :return: ``pandas`` ``Series`` or ``DataFrame``; rescaled data
    """

    return df/df.std(axis=0)


def add_dv(name, y):
    """

    :param name: ``str``; name of dependent variable
    :param y: ``pandas`` ``DataFrame``; response data.
    :return: ``pandas`` ``DataFrame``; response data with any missing ops applied to DV.
    """

    if name in y.columns:
        return y

    op, var = op_finder.match(name).groups()

    y = add_dv(var, y)
    arr = y[var]

    if op in ['c', 'c.']:
        new_col = c(arr)
    elif op in ['z', 'z.']:
        new_col = z(arr)
    elif op in ['s', 's.']:
        new_col = s(arr)
    elif op == 'log':
        new_col = np.log(arr)
    elif op == 'log1p':
        new_col = np.log(arr + 1)
    elif op == 'exp':
        new_col = np.exp(arr)
    else:
        raise ValueError('Unrecognized op: "%s".' % op)

    y[name] = new_col

    return y


def get_first_last_obs_lists(y):
    """
    Convenience utility to extract out all first_obs and last_obs columns in **y** sorted by file index

    :param y: ``pandas`` ``DataFrame``; response data.
    :return: pair of ``list`` of ``str``; first_obs column names and last_obs column names
    """

    first_obs = []
    last_obs = []
    last_obs_ix = [int(c.split('_')[-1]) for c in y.columns if c.startswith('last_obs')]
    for i in sorted(last_obs_ix):
        first_obs_cur = y['first_obs_%d' % i]
        last_obs_cur = y['last_obs_%d' % i]
        first_obs.append(first_obs_cur)
        last_obs.append(last_obs_cur)

    return first_obs, last_obs


def filter_invalid_responses(y, dv):
    """
    Filter out rows with non-finite responses.

    :param y: ``pandas`` ``DataFrame``; response data.
    :param dv: ``str``; name of column containing the dependent variable
    :return: 2-tuple of ``pandas`` ``DataFrame`` and ``pandas`` ``Series``; valid data and indicator vector used to filter out invalid data.
    """

    first_obs, last_obs = get_first_last_obs_lists(y)

    select_y_valid = np.isfinite(y[dv])
    
    return y[select_y_valid], select_y_valid


def build_DTSR_impulses(
        X,
        first_obs,
        last_obs,
        impulse_names,
        time_y=None,
        history_length=128,
        X_response_aligned_predictor_names=None,
        X_response_aligned_predictors=None,
        X_2d_predictor_names=None,
        X_2d_predictors=None,
        int_type='int32',
        float_type='float32',
):
    """
    Construct 3D array of shape ``(batch_len, history_length, n_impulses)`` to use as predictor data for DTSR fitting.

    :param X: list of ``pandas`` ``DataFrame``; impulse (predictor) data.
    :param first_obs: list of ``pandas`` ``Series``; vector of row indices in **X** of the first impulse in the time series associated with each response.
    :param last_obs: list of ``pandas`` ``Series``; vector of row indices in **X** of the last preceding impulse in the time series associated with each response.
    :param impulse_names: ``list`` of ``str``; names of columns in **X** to be used as impulses by the model.
    :param time_y: ``numpy`` 1D array; vector of response timestamps. Needed to timestamp any response-aligned predictors (ignored if none in model).
    :param history_length: ``int``; maximum number of history observations.
    :param X_response_aligned_predictor_names: ``list`` of ``str``; names of predictors measured synchronously with the response rather than the impulses. If ``None``, no such impulses.
    :param X_response_aligned_predictors: ``pandas`` ``DataFrame`` or ``None``; table of predictors measured synchronously with the response rather than the impulses. If ``None``, no such impulses.
    :param X_2d_predictor_names: ``list`` of ``str``; names of 2D impulses (impulses whose value depends on properties of the most recent impulse). If ``None``, no such impulses.
    :param X_2d_predictors: ``pandas`` ``DataFrame`` or ``None``; table of 2D impulses. If ``None``, no such impulses.
    :param int_type: ``str``; name of int type.
    :param float_type: ``str``; name of float type.
    :return: 3-tuple of ``numpy`` arrays; the expanded impulse array, the expanded timestamp array, and a boolean mask zeroing out locations of non-existent impulses.
    """

    if not isinstance(X, list):
        X = [X]
    if not isinstance(first_obs, list):
        first_obs = [first_obs]
    if not isinstance(last_obs, list):
        last_obs = [last_obs]

    if X_response_aligned_predictor_names is None:
        X_response_aligned_predictor_names = []
    assert (X_2d_predictors is None and (X_2d_predictor_names is None or len(X_2d_predictor_names) == 0)) or (X_2d_predictors.shape[-1] == len(X_2d_predictor_names)), 'Shape mismatch between X_2d_predictors and X_2d_predictor_names'
    if X_2d_predictor_names is None:
        X_2d_predictor_names = []

    impulse_names_1d = sorted(list(set(impulse_names).difference(set(X_response_aligned_predictor_names)).difference(set(X_2d_predictor_names))))
    impulse_names_1d_todo = set(impulse_names_1d)
    impulse_names_1d_tmp = []

    X_2d_from_1d = []
    time_X_2d = []
    time_mask = []
    for i, X_cur in enumerate(X):
        impulse_names_1d_cur = impulse_names_1d_todo.intersection(set(X_cur.columns))
        if len(impulse_names_1d_cur) > 0:
            impulse_names_1d_todo = impulse_names_1d_todo - impulse_names_1d_cur
            impulse_names_1d_cur = sorted(list(impulse_names_1d_cur))
            impulse_names_1d_tmp += impulse_names_1d_cur
            X_2d_from_1d_cur, time_X_2d_cur, time_mask_cur = expand_history(
                X_cur[impulse_names_1d_cur],
                X_cur.time,
                first_obs[i],
                last_obs[i],
                history_length,
                int_type=int_type,
                float_type=float_type
            )
            X_2d_from_1d.append(X_2d_from_1d_cur)
            time_X_2d.append(time_X_2d_cur)
            time_mask.append(time_mask_cur)

    assert len(impulse_names_1d_todo) == 0, 'Not all impulses were processed during DTSR data array construction. Remaining impulses: %s' % impulse_names_1d_todo
    impulse_names_1d = impulse_names_1d_tmp

    X_2d = np.concatenate(X_2d_from_1d, axis=-1)
    X_2d = X_2d[:,:,names2ix(impulse_names_1d, impulse_names_1d_tmp)]
    time_X_2d = np.concatenate(time_X_2d, axis=-1)
    time_mask = np.concatenate(time_mask, axis=-1)

    if X_response_aligned_predictors is not None:
        response_aligned_shape = (X_2d.shape[0], X_2d.shape[1], len(X_response_aligned_predictor_names))
        X_response_aligned_predictors_new = np.zeros(response_aligned_shape)
        X_response_aligned_predictors_new[:, -1, :] = X_response_aligned_predictors[X_response_aligned_predictor_names]
        X_2d = np.concatenate([X_2d, X_response_aligned_predictors_new], axis=2)

        time_X_2d_new = np.zeros(response_aligned_shape)
        time_X_2d_new[:,-1,:] = time_y[..., None]
        time_X_2d = np.concatenate([time_X_2d, time_X_2d_new], axis=2)

        time_mask_new = np.zeros(response_aligned_shape)
        time_mask_new[:,-1,:] = 1.
        time_mask = np.concatenate([time_mask, time_mask_new], axis=2)

    if X_2d_predictors is not None:
        raise ValueError('2D predictors are currently fatally bugged. Do not use them.')
        X_2d = np.concatenate([X_2d, X_2d_predictors], axis=2)

    # Ensure that impulses are properly aligned
    impulse_names_cur = impulse_names_1d + X_response_aligned_predictor_names + X_2d_predictor_names
    ix = names2ix(impulse_names, impulse_names_cur)
    X_2d = X_2d[:,:,ix]
    time_X_2d = time_X_2d[:,:,ix]
    time_mask = time_mask[:,:,ix]

    return X_2d, time_X_2d, time_mask


def compute_history_intervals(X, y, series_ids):
    """
    Compute row indices in **X** of initial and final impulses for each element of **y**.

    :param X: ``pandas`` ``DataFrame``; impulse (predictor) data.
    :param y: ``pandas`` ``DataFrame``; response data.
    :param series_ids: ``list`` of ``str``; column names whose jointly unique values define unique time series.
    :return: 2-tuple of ``numpy`` vectors; first and last impulse observations (respectively) for each response in **y**
    """

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
        if i == 0 or i % 1000 == 999 or i == n-1:
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


def corr_dtsr(X_2d, impulse_names, impulse_names_2d, time, time_mask):
    """
    Compute correlation matrix, including correlations across time where necessitated by 2D predictors.

    :param X_2d: ``numpy`` array; the impulse data. Must be of shape ``(batch_len, history_length, n_impulses)``, can be computed from sources by ``build_DTSR_impulses()``.
    :param impulse_names: ``list`` of ``str``; names of columns in **X_2d** to be used as impulses by the model.
    :param impulse_names_2d: ``list`` of ``str``; names of columns in **X_2d** that designate to 2D predictors.
    :param time: 3D ``numpy`` array; array of timestamps for each event in **X_2d**.
    :param time_mask: 3D ``numpy`` array; array of masks over padding events in **X_2d**.
    :return: ``pandas`` ``DataFrame``; the correlation matrix.
    """

    rho = pd.DataFrame(np.zeros((len(impulse_names), len(impulse_names))), index=impulse_names, columns=impulse_names)

    for i in range(len(impulse_names)):
        for j in range(i, len(impulse_names)):
            if impulse_names[i] in impulse_names_2d or impulse_names[j] in impulse_names_2d:
                x1 = X_2d[..., i]
                x2 = X_2d[..., j]

                aligned = np.logical_and(np.logical_and(np.isclose(time[:,:,i], time[:,:,j]), time_mask[:,:,i]), time_mask[:,:,j])
                n = aligned.sum()
                x1_mean = x1.sum() / n
                x2_mean = x2.sum() / n
                cor = ((x1 - x1_mean) * (x2 - x2_mean) * aligned).sum() / \
                      np.sqrt(((x1 - x1_mean) ** 2 * aligned).sum() * (
                              (x2 - x2_mean) ** 2 * aligned).sum())
            else:
                x1 = X_2d[:, -1, i]
                x2 = X_2d[:, -1, j]

                n = X_2d.shape[0]
                x1_mean = x1.sum() / n
                x2_mean = x2.sum() / n
                cor = ((x1 - x1_mean) * (x2 - x2_mean)).sum() / \
                      np.sqrt(((x1 - x1_mean) ** 2).sum() * ((x2 - x2_mean) ** 2).sum())

            rho.loc[impulse_names[i], impulse_names[j]] = cor
            if i != j:
                rho.loc[impulse_names[j], impulse_names[i]] = cor

    return rho


def compute_filters(y, filter_map=None):
    """
    Compute filters given a filter map.

    :param y: ``pandas`` ``DataFrame``; response data.
    :param filter_map: ``dict``; maps column names to filtering criteria for their values.
    :return: ``numpy`` vector; boolean mask to use for ``pandas`` subsetting operations.
    """

    if filter_map is None:
        return y
    select = np.ones(len(y), dtype=bool)
    for field in filter_map:
        if field in y.columns:
            for cond in filter_map[field]:
                select &= compute_filter(y, field, cond)
    return select


def compute_filter(y, field, cond):
    """
    Compute filter given a field and condition

    :param y: ``pandas`` ``DataFrame``; response data.
    :param field: ``str``; name of column on whose values to filter.
    :param cond: ``str``; string representation of condition to use for filtering.
    :return: ``numpy`` vector; boolean mask to use for ``pandas`` subsetting operations.
    """

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
    """
    Map tuples in columns designated by **split_fields** into integer ID to use for data partitioning.

    :param y: ``pandas`` ``DataFrame``; response data.
    :param split_fields: ``list`` of ``str``; column names to use for computing split ID.
    :return: ``numpy`` vector; integer vector of split ID's.
    """

    splitID = np.zeros(len(y), dtype='int32')
    for col in split_fields:
        splitID += y[col].cat.codes
    return splitID


def compute_partition(y, modulus, n):
    """
    Given a ``splitID`` column, use modular arithmetic to partition data into **n** subparts.

    :param y: ``pandas`` ``DataFrame``; response data.
    :param modulus: ``int``; modulus to use for splitting, must be at least as large as **n**.
    :param n: ``int``; number of subparts in the partition.
    :return: ``list`` of ``numpy`` vectors; one boolean vector per subpart of the partition, selecting only those elements of **y** that belong.
    """

    partition = [((y.splitID) % modulus) <= (modulus - n)]
    for i in range(n-1, 0, -1):
        partition.append(((y.splitID) % modulus) == (modulus - i))
    return partition


def expand_history(X, X_time, first_obs, last_obs, history_length, int_type='int32', float_type='float32', fill=0.):
    """
    Expand out impulse stream in **X** for each response in the target data.

    :param X: ``pandas`` ``DataFrame``; impulse (predictor) data.
    :param X_time: ``pandas`` ``Series``; timestamps associated with each impulse in **X**.
    :param first_obs: ``pandas`` ``Series``; vector of row indices in **X** of the first impulse in the time series associated with each response.
    :param last_obs: ``pandas`` ``Series``; vector of row indices in **X** of the last preceding impulse in the time series associated with each response.
    :param history_length: ``int``; maximum number of history observations.
    :param int_type: ``str``; name of int type.
    :param float_type: ``str``; name of float type.
    :param fill: ``float``; fill value for padding cells.
    :return: 3-tuple of ``numpy`` arrays; the expanded impulse array, the expanded timestamp array, and a boolean mask zeroing out locations of non-existent impulses.
    """

    INT_NP = getattr(np, int_type)
    FLOAT_NP = getattr(np, float_type)
    last_obs = np.array(last_obs, dtype=INT_NP)
    first_obs = np.maximum(np.array(first_obs, dtype=INT_NP), last_obs - history_length)
    X_time = np.array(X_time, dtype=FLOAT_NP)
    X = np.array(X)

    X_2d = np.full((first_obs.shape[0], history_length, X.shape[1]), fill, dtype=FLOAT_NP)
    time_X_2d = np.zeros_like(X_2d)
    time_mask = np.zeros_like(X_2d)

    for i, first, last in zip(np.arange(first_obs.shape[0]), first_obs, last_obs):
        if first < last:
            sX = X[first:last]
            sXt = X_time[first:last]
            X_2d[i, -sX.shape[0]:] = sX
            time_X_2d[i][-len(sXt):] = sXt[..., None]
            time_mask[i][-len(sXt):] = 1

    return X_2d, time_X_2d, time_mask


def compute_time_mask(X_time, first_obs, last_obs, history_length, int_type='int32', float_type='float32'):
    """
    Compute mask for expanded impulse data zeroing out non-existent impulses.

    :param X_time: ``pandas`` ``Series``; timestamps associated with each impulse in **X**.
    :param first_obs: ``pandas`` ``Series``; vector of row indices in **X** of the first impulse in the time series associated with each response.
    :param last_obs: ``pandas`` ``Series``; vector of row indices in **X** of the last preceding impulse in the time series associated with each response.
    :param history_length: ``int``; maximum number of history observations.
    :param int_type: ``str``; name of int type.
    :param float_type: ``str``; name of float type.
    :return: ``numpy`` array; boolean impulse mask.
    """

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


def preprocess_data(X, y, formula_list, series_ids, filter_map=None, compute_history=True, history_length=128, debug=False):
    """
    Preprocess DTSR data.

    :param X: list of ``pandas`` ``DataFrame``; impulse (predictor) data.
    :param y: ``pandas`` ``DataFrame``; response data.
    :param formula_list: ``list`` of ``Formula``; DTSR formula for which to preprocess data.
    :param series_ids: ``list`` of ``str``; column names whose jointly unique values define unique time series.
    :param filter_map: ``dict``; map from column names to string representations of filters to apply using those columns' values. See the DTSR config file reference for details.
    :param compute_history: ``bool``; compute history intervals for each regression target.
    :param history_length: ``int``; maximum number of history observations.
    :param debug: ``bool``; print debugging information
    :return: 7-tuple; predictor data, response data, filtering mask, response-aligned predictor names, response-aligned predictors, 2D predictor names, and 2D predictors
    """

    sys.stderr.write('Pre-processing data...\n')

    if not isinstance(X, list):
        X = [X]

    if filter_map is None:
        select = np.full((len(y),), True, dtype='bool')
    else:
        select = compute_filters(y, filter_map)
        y = y[select]

    X_response_aligned_predictor_names = None
    X_response_aligned_predictors = None
    X_2d_predictor_names = None
    X_2d_predictors = None

    if compute_history:
        X_new = []
        for i in range(len(X)):
            X_cur = X[i]
            sys.stderr.write('Computing history intervals for each regression target in predictor file %d...\n' % (i+1))
            first_obs, last_obs = compute_history_intervals(X_cur, y, series_ids)
            y['first_obs_%d' % i] = first_obs
            y['last_obs_%d' % i] = last_obs

            # Floating point precision issues can allow the response to precede the impulse for simultaneous X/y,
            # which can break downstream convolution. The correction below to y.time prevents this.
            y.time = np.where(last_obs > first_obs, np.maximum(np.array(X_cur.time)[last_obs - 1], y.time), y.time)

            if debug:
                sample = np.random.randint(0, len(y), 10)
                sample = np.concatenate([np.zeros((1,), dtype='int'), sample, np.ones((1,), dtype='int') * (len(y)-1)], axis=0)
                for i in sample:
                    print(i)
                    row = y.iloc[i]
                    print(row[['subject', 'docid', 'time']])
                    print(X_cur[['subject', 'docid', 'word', 'time']][row.first_obs:row.last_obs])
            X_new.append(X_cur)

        for x in formula_list:
            X_new, y, X_response_aligned_predictor_names, X_response_aligned_predictors, X_2d_predictor_names, X_2d_predictors = x.apply_formula(
                X_new,
                y,
                X_2d_predictor_names=X_2d_predictor_names,
                X_2d_predictors=X_2d_predictors,
                X_response_aligned_predictor_names=X_response_aligned_predictor_names,
                X_response_aligned_predictors=X_response_aligned_predictors,
                history_length=history_length
            )
    else:
        X_new = X

    return X_new, y, select, X_response_aligned_predictor_names, X_response_aligned_predictors, X_2d_predictor_names, X_2d_predictors

