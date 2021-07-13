import re
import numpy as np
import pandas as pd
from .util import names2ix, stderr

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


def corr(A, B):
    # Assumes A and B are n x a and n x b matrices and computes a x b pairwise correlations
    A_centered = A - A.mean(axis=0, keepdims=True)
    B_centered = B - B.mean(axis=0, keepdims=True)

    A_ss = (A_centered ** 2).sum(axis=0)
    B_ss = (B_centered ** 2).sum(axis=0)

    rho = np.dot(A_centered.T, B_centered) / np.sqrt(np.dot(A_ss[..., None], B_ss[None, ...]))
    rho = np.clip(rho, -1, 1)
    return rho


def corr_cdr(X_2d, impulse_names, impulse_names_2d, time, time_mask):
    """
    Compute correlation matrix, including correlations across time where necessitated by 2D predictors.

    :param X_2d: ``numpy`` array; the impulse data. Must be of shape ``(batch_len, history_length+future_length, n_impulses)``, can be computed from sources by ``build_CDR_impulses()``.
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


def add_responses(names, y):
    """
    Add response variable(s) to a dataframe, applying any preprocessing required by the formula string.

    :param names: ``str`` or ``list`` of ``str``; name(s) of dependent variable(s)
    :param y: ``pandas`` ``DataFrame``; response data.
    :return: ``pandas`` ``DataFrame``; response data with any missing ops applied.
    """

    if isinstance(names, str):
        names = [names]

    for name in names:
        if name in y.columns:
            return y

        op, var = op_finder.match(name).groups()

        y = add_responses(var, y)
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
    Convenience utility to extract out all first_obs and last_obs columns in **Y** sorted by file index

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


def filter_invalid_responses(Y, dv, crossval_factor=None, crossval_fold=None):
    """
    Filter out rows with non-finite responses.

    :param Y: ``pandas`` table or ``list`` of ``pandas`` tables; response data.
    :param dv: ``str`` or ``list`` of ``str``; name(s) of column(s) containing the dependent variable(s)
    :param crossval_factor: ``str`` or ``None``; name of column containing the selection variable for cross validation. If ``None``, no cross validation filtering.
    :param crossval_fold: ``list`` or ``None``; list of valid values for cross-validation selection. Used only if ``crossval_factor`` is not ``None``.
    :return: 2-tuple of ``pandas`` ``DataFrame`` and ``pandas`` ``Series``; valid data and indicator vector used to filter out invalid data.
    """

    df_in = False
    if not isinstance(Y, list):
        Y = [Y]
        df_in = True

    if not isinstance(dv, list):
        dv = [dv]

    if crossval_fold is None:
        crossval_fold = []

    select_Y_valid = []
    for i, _Y in enumerate(Y):
        _select_Y_valid = np.ones(len(_Y), dtype=bool)
        if crossval_factor:
            _select_Y_valid &= _Y[crossval_factor].isin(crossval_fold)
        for _dv in dv:
            if _dv in _Y:
                dtype = _Y[_dv].dtype
                if dtype.name != 'category' and np.issubdtype(dtype, np.number):
                    is_numeric = True
                else:
                    is_numeric = False
                if is_numeric:
                    _select_Y_valid &= np.isfinite(_Y[_dv])
        select_Y_valid.append(_select_Y_valid)
        Y[i] = _Y[_select_Y_valid]

    if df_in:
        Y = Y[0]
        select_Y_valid = select_Y_valid[0]
    
    return Y, select_Y_valid


def build_CDR_data_inner(
        X,
        Y=None,
        first_obs=None,
        last_obs=None,
        Y_time=None,
        Y_category_map=None,
        impulse_names=None,
        response_names=None,
        history_length=128,
        future_length=0,
        X_response_aligned_predictor_names=None,
        X_response_aligned_predictors=None,
        X_2d_predictor_names=None,
        X_2d_predictors=None,
        int_type='int32',
        float_type='float32',
):
    """
    Construct data arrays in the required format for CDR fitting/evaluation for a single response array.

    :param X: ``list`` of ``pandas`` tables; impulse (predictor) data.
    :param Y: ``pandas`` table; response data. If ``None``, does not return a response array.
    :param first_obs: ``list`` of index vectors (``list``, ``pandas`` series, or ``numpy`` vector) of first observations; the list contains vectors of row indices, one for each element of **X**, of the first impulse in the time series associated with the response. If ``None``, inferred from **Y**.
    :param last_obs: ``list`` of index vectors (``list``, ``pandas`` series, or ``numpy`` vector) of last observations; the list contains vectors of row indices, one for each element of **X**, of the last impulse in the time series associated with the response. If ``None``, inferred from **Y**.
    :param Y_time: `response timestamp vector (``list``, ``pandas`` series, or ``numpy`` vector); vector of response timestamps. Needed to timestamp any response-aligned predictors (ignored if none in model).
    :param Y_category_map: ``dict`` or ``None``; map from category labels to integers for each categorical response.
    :param impulse_names: ``list`` of ``str``; names of columns in **X** to be used as impulses by the model. If ``None``, all columns returned.
    :param response_names: ``list`` of ``str``; names of columns in **Y** to be used as responses by the model. If ``None``, all columns returned.
    :param history_length: ``int``; maximum number of history (backward) observations.
    :param future_length: ``int``; maximum number of future (forward) observations.
    :param X_response_aligned_predictor_names: ``list`` of ``str``; names of predictors measured synchronously with the response rather than the impulses. If ``None``, no such impulses.
    :param X_response_aligned_predictors: ``pandas`` ``DataFrame`` or ``None``; table of predictors measured synchronously with the response rather than the impulses. If ``None``, no such impulses.
    :param X_2d_predictor_names: ``list`` of ``str``; names of 2D impulses (impulses whose value depends on properties of the most recent impulse). If ``None``, no such impulses.
    :param X_2d_predictors: ``pandas`` ``DataFrame`` or ``None``; table of 2D impulses. If ``None``, no such impulses.
    :param int_type: ``str``; name of int type.
    :param float_type: ``str``; name of float type.
    :return: 6-tuple of ``numpy`` arrays; let N, T, I, R respectively be the number of rows in **Y**, history length, number of impulse dimensions, and number of response dimensions. Outputs are (1) impulses with shape (N, T, I), (2) impulse timestamps with shape (N, T, I), impulse mask with shape (N, T, I), responses with shape (N, R) or ``None`` if **Y** is ``None``, response timestamps with shape (N, R) or ``None`` if **Y** is ``None``, and response masks with shape (N, R) or ``None`` if **Y** is ``None``.
    """

    # Check prerequisites
    assert isinstance(X, list) and not [_X for _X in X if not isinstance(_X, pd.DataFrame)], "X must be a list of pandas DataFrames"
    assert Y is None or isinstance(Y, pd.DataFrame), "Y must either be ``None`` or a pandas DataFrame"
    assert first_obs is None or isinstance(first_obs, list), "first_obs must either be ``None`` or a list"
    assert last_obs is None or isinstance(last_obs, list), "last_obs must either be ``None`` or a list"
    assert (Y is not None) or (first_obs is not None and last_obs is not None and Y_time is not None), "If Y is not provided, first_obs, last_obs, and time_y must be provided."

    if first_obs is None:
        first_obs = [Y['first_obs_%d' % i] for i in range(len(X))]

    if last_obs is None:
        last_obs = [Y['last_obs_%d' % i] for i in range(len(X))]

    if Y_time is None:
        Y_time = Y.time

    if X_response_aligned_predictor_names is None:
        X_response_aligned_predictor_names = []
    assert (X_2d_predictors is None and (X_2d_predictor_names is None or len(X_2d_predictor_names) == 0)) or (X_2d_predictors.shape[-1] == len(X_2d_predictor_names)), 'Shape mismatch between X_2d_predictors and X_2d_predictor_names'
    if X_2d_predictor_names is None:
        X_2d_predictor_names = []

    if not (impulse_names):  # Empty (intercept-only) model
        intercept_only = True
        impulse_names = ['time']
    else:
        intercept_only = False

    if Y_category_map is None:
        Y_category_map = {}

    # Process impulses

    impulse_names_1d = sorted(list(set(impulse_names).difference(set(X_response_aligned_predictor_names)).difference(set(X_2d_predictor_names))))
    impulse_names_1d_todo = set(impulse_names_1d)
    impulse_names_1d_tmp = []

    X_2d_from_1d = []
    X_time_2d = []
    X_mask = []

    window_length = history_length + future_length

    for i, _X in enumerate(X):
        impulse_names_1d_cur = impulse_names_1d_todo.intersection(set(_X.columns))
        if len(impulse_names_1d_cur) > 0:
            impulse_names_1d_todo = impulse_names_1d_todo - impulse_names_1d_cur
            impulse_names_1d_cur = sorted(list(impulse_names_1d_cur))
            impulse_names_1d_tmp += impulse_names_1d_cur
            _X_2d_from_1d, _X_time_2d, _X_mask = expand_impulse_sequence(
                _X[impulse_names_1d_cur],
                _X.time,
                first_obs[i],
                last_obs[i],
                window_length,
                int_type=int_type,
                float_type=float_type
            )
            X_2d_from_1d.append(_X_2d_from_1d)
            X_time_2d.append(_X_time_2d)
            X_mask.append(_X_mask)

    assert len(impulse_names_1d_todo) == 0, 'Not all impulses were processed during CDR data array construction. Remaining impulses: %s' % impulse_names_1d_todo
    impulse_names_1d = impulse_names_1d_tmp

    X_2d = np.concatenate(X_2d_from_1d, axis=-1)
    X_2d = X_2d[:,:,names2ix(impulse_names_1d, impulse_names_1d_tmp)]
    X_time_2d = np.concatenate(X_time_2d, axis=-1)
    X_mask = np.concatenate(X_mask, axis=-1)

    if X_response_aligned_predictors is not None:
        response_aligned_shape = (X_2d.shape[0], X_2d.shape[1], len(X_response_aligned_predictor_names))
        X_response_aligned_predictors_new = np.zeros(response_aligned_shape)
        X_response_aligned_predictors_new[:, -1, :] = X_response_aligned_predictors[X_response_aligned_predictor_names]
        X_2d = np.concatenate([X_2d, X_response_aligned_predictors_new], axis=2)

        time_X_2d_new = np.zeros(response_aligned_shape)
        time_X_2d_new[:,-1,:] = np.array(Y_time)[..., None]
        X_time_2d = np.concatenate([X_time_2d, time_X_2d_new], axis=2)

        time_mask_new = np.zeros(response_aligned_shape)
        time_mask_new[:,-1,:] = 1.
        X_mask = np.concatenate([X_mask, time_mask_new], axis=2)

    if X_2d_predictors is not None:
        raise ValueError('2D predictors are currently broken. Do not use them.')
        X_2d = np.concatenate([X_2d, X_2d_predictors], axis=2)

    # Ensure that impulses are properly aligned
    impulse_names_cur = impulse_names_1d + X_response_aligned_predictor_names + X_2d_predictor_names
    ix = names2ix(impulse_names, impulse_names_cur)
    X_2d = X_2d[:,:,ix]
    X_time_2d = X_time_2d[:,:,ix]
    X_mask = X_mask[:,:,ix]

    if intercept_only:
        X_2d = X_2d[..., 0:0]

    # Process responses

    if Y is None:
        Y_out = None
        Y_mask = None
    else:
        Y_out = []
        Y_mask = []

        if response_names is None:
            response_names = list(Y.columns)
        for name in response_names:
            if name in Y:
                _Y_dv = Y[name]
                if name in Y_category_map:
                    _Y_dv = _Y_dv.map(lambda x: Y_category_map[name].get(x, x))
                Y_out.append(_Y_dv)
                Y_mask.append(np.ones(len(Y)))
            else:
                Y_out.append(np.zeros(len(Y)))
                Y_mask.append(np.zeros(len(Y)))

        Y_out = np.stack(Y_out, axis=1)
        Y_mask = np.stack(Y_mask, axis=1)

    return X_2d, X_time_2d, X_mask, Y_out, Y_mask


def build_CDR_data(
        X,
        Y=None,
        first_obs=None,
        last_obs=None,
        Y_time=None,
        Y_category_map=None,
        impulse_names=None,
        response_names=None,
        history_length=128,
        future_length=0,
        X_response_aligned_predictor_names=None,
        X_response_aligned_predictors=None,
        X_2d_predictor_names=None,
        X_2d_predictors=None,
        int_type='int32',
        float_type='float32',
):
    """
    Construct data arrays in the required format for CDR fitting/evaluation for one or more response arrays.

    :param X: ``list`` of ``pandas`` tables; impulse (predictor) data.
    :param Y: ``list`` of ``pandas`` tables; response data. If ``None``, does not return a response array.
    :param first_obs: ``list`` of ``list`` of index vectors (``list``, ``pandas`` series, or ``numpy`` vector) of first observations; the list contains one element for each response array. Inner lists contain vectors of row indices, one for each element of **X**, of the first impulse in the time series associated with each response. If ``None``, inferred from **Y**.
    :param last_obs: ``list`` of ``list`` of index vectors (``list``, ``pandas`` series, or ``numpy`` vector) of last observations; the list contains one element for each response array. Inner lists contain vectors of row indices, one for each element of **X**, of the last impulse in the time series associated with each response. If ``None``, inferred from **Y**.
    :param Y_time: ``list`` of response timestamp vectors (``list``, ``pandas`` series, or ``numpy`` vector); vector(s) of response timestamps, one for each response array. Needed to timestamp any response-aligned predictors (ignored if none in model).
    :param Y_category_map: ``dict`` or ``None``; map from category labels to integers for each categorical response.
    :param impulse_names: ``list`` of ``str``; names of columns in **X** to be used as impulses by the model. If ``None``, all columns returned.
    :param response_names: ``list`` of ``str``; names of columns in **Y** to be used as responses by the model. If ``None``, all columns returned.
    :param history_length: ``int``; maximum number of history (backward) observations.
    :param future_length: ``int``; maximum number of future (forward) observations.
    :param X_response_aligned_predictor_names: ``list`` of ``str``; names of predictors measured synchronously with the response rather than the impulses. If ``None``, no such impulses.
    :param X_response_aligned_predictors: ``pandas`` ``DataFrame`` or ``None``; table of predictors measured synchronously with the response rather than the impulses. If ``None``, no such impulses.
    :param X_2d_predictor_names: ``list`` of ``str``; names of 2D impulses (impulses whose value depends on properties of the most recent impulse). If ``None``, no such impulses.
    :param X_2d_predictors: ``pandas`` ``DataFrame`` or ``None``; table of 2D impulses. If ``None``, no such impulses.
    :param int_type: ``str``; name of int type.
    :param float_type: ``str``; name of float type.
    :return: 6-tuple of ``numpy`` arrays; let N, T, I, R respectively be the number of rows (total number of rows in **Y**), history length, number of impulse dimensions, and number of response dimensions. Outputs are (1) impulses with shape (N, T, I), (2) impulse timestamps with shape (N, T, I), impulse mask with shape (N, T, I), responses with shape (N, R) or ``None`` if **Y** is ``None``, response timestamps with shape (N, R) or ``None`` if **Y** is ``None``, and response masks with shape (N, R) or ``None`` if **Y** is ``None``.
    """

    # Check prerequisites
    assert isinstance(X, list), "X must be a list"
    assert Y is None or isinstance(Y, list), "Y must either be ``None`` or a list"
    assert first_obs is None or isinstance(first_obs, list) and not [x for x in first_obs if not isinstance(x, list)], "first_obs must either be ``None`` or a list of lists"
    assert last_obs is None or isinstance(last_obs, list) and not [x for x in last_obs if not isinstance(x, list)], "last_obs must either be ``None`` or a list of lists"
    assert Y_time is None or isinstance(Y_time, list), "Y_time must either be ``None`` or a list"
    assert (Y is not None) or (first_obs is not None and last_obs is not None and Y_time is not None), "If Y is not provided, first_obs, last_obs, and time_y must be provided."

    X_2d = []
    X_time_2d = []
    X_mask = []

    if Y is None:
        Y_out = None
        Y_time_out = None
        Y_mask_out = None
        Y_n_tables = len(first_obs)
    else:
        Y_out = []
        Y_time_out = []
        Y_mask_out = []
        Y_n_tables = len(Y)

    for i in range(Y_n_tables):
        if Y is None:
            _Y = Y
        else:
            _Y = Y[i]

        if first_obs is None:
            _first_obs = [_Y['first_obs_%d' % j] for j in range(len(X))]
        else:
            _first_obs = first_obs[i]
            
        if last_obs is None:
            _last_obs = [_Y['last_obs_%d' % j] for j in range(len(X))]
        else:
            _last_obs = last_obs[i]
    
        if Y_time is None:
            _Y_time = _Y.time
        else:
            _Y_time = Y_time[i]

        _X_2d, _X_time_2d, _X_mask, _Y_out, _Y_mask = build_CDR_data_inner(
            X,
            Y=_Y,
            first_obs=_first_obs,
            last_obs=_last_obs,
            Y_time=_Y_time,
            Y_category_map=Y_category_map,
            impulse_names=impulse_names,
            response_names=response_names,
            history_length=history_length,
            future_length=future_length,
            X_response_aligned_predictor_names=X_response_aligned_predictor_names,
            X_response_aligned_predictors=None if X_response_aligned_predictors is None else X_response_aligned_predictors[i],
            X_2d_predictor_names=X_2d_predictor_names,
            X_2d_predictors=X_2d_predictors,
            int_type=int_type,
            float_type=float_type,
        )

        X_2d.append(_X_2d)
        X_time_2d.append(_X_time_2d)
        X_mask.append(_X_mask)

        if _Y is not None:
            Y_out.append(_Y_out)
            Y_time_out.append(_Y_time)
            Y_mask_out.append(_Y_mask)

    X_2d = np.concatenate(X_2d, axis=0)
    X_time_2d = np.concatenate(X_time_2d, axis=0)
    X_mask = np.concatenate(X_mask, axis=0)

    if Y is not None:
        Y_out = np.concatenate(Y_out, axis=0)
        Y_time_out = np.concatenate(Y_time_out, axis=0)
        Y_mask_out = np.concatenate(Y_mask_out, axis=0)

    return X_2d, X_time_2d, X_mask, Y_out, Y_time_out, Y_mask_out


def get_rangf_array(
        Y,
        rangf_names,
        rangf_map
):
    """
    Collect random grouping factor indicators as ``numpy`` integer arrays that can be read by Tensorflow.
    Returns vertical concatenation of GF arrays from each element of **Y**.

    :param Y: ``pandas`` table or ``list`` of ``pandas`` tables; response data.
    :param rangf_names: ``list`` of ``str``; names of columns containing random grouping factor levels (order is preserved, changing the order will change the resulting array).
    :param rangf_map: ``list`` of ``dict``; map for each random grouping factor from levels to unique indices.
    :return:
    """

    if not isinstance(Y, list):
        Y = [Y]

    Y_rangf = []

    for _Y in Y:
        _Y_rangf = _Y[rangf_names]
        for i in range(len(rangf_names)):
            c = rangf_names[i]
            _Y_rangf[c] = pd.Series(_Y_rangf[c].astype(str)).map(rangf_map[i])
        _Y_rangf = np.array(_Y_rangf, dtype=int)
        Y_rangf.append(_Y_rangf)

    Y_rangf = np.concatenate(Y_rangf, axis=0)

    return Y_rangf


def get_time_windows(
        X,
        Y,
        series_ids,
        forward=False,
        window_length=128,
        exclude_instantaneous=False,
        verbose=True
):
    """
    Compute row indices in **X** of initial and final impulses for each element of **y**.
    Assumes time series are already sorted by **series_ids**.

    :param X: ``pandas`` ``DataFrame``; impulse (predictor) data.
    :param Y: ``pandas`` ``DataFrame``; response data.
    :param series_ids: ``list`` of ``str``; column names whose jointly unique values define unique time series.
    :param forward: ``bool``; whether to compute forward windows (future inputs) or backward windows (past inputs, used if **forward** is ``False``).
    :param window_length: ``int``; maximum size of time window to consider. If ``np.inf``, no bound on window size.
    :param exclude_instantaneous: ``bool``; whether to exclude impulses that have the same timestamp as their targets.
    :param verbose: ``bool``; whether to report progress to stderr
    :return: 2-tuple of ``numpy`` vectors; first and last impulse observations (respectively) for each response in **y**
    """

    if window_length is None:
        window_length = 0

    m = len(X)
    n = len(Y)

    X_src = X
    Y_src = Y

    if forward: # Reverse the time dimension
        X = X[series_ids + ['time']].reset_index(drop=True)
        X['time'] = -X['time']
        X = X.sort_values(series_ids + ['time'])
        X_ix = np.concatenate([np.array(X.index), [len(X)]], axis=0) # Pad 1 to handle final interval
        Y = Y[series_ids + ['time']].reset_index(drop=True)
        Y['time'] = -Y['time']
        Y = Y.sort_values(series_ids + ['time'])
        Y_ix = np.array(Y.index)
    else:
        X_ix = np.arange(len(X)+1) # Pad 1 to handle final interval
        Y_ix = np.arange(len(Y))

    X_time = np.array(X.time)
    Y_time = np.array(Y.time)

    X_id_vectors = []
    Y_id_vectors = []

    for i in range(len(series_ids)):
        col = series_ids[i]
        X_id_vectors.append(np.array(X[col]))
        Y_id_vectors.append(np.array(Y[col]))
    X_id_vectors = np.stack(X_id_vectors, axis=1)
    Y_id_vectors = np.stack(Y_id_vectors, axis=1)

    Y_id = Y_id_vectors[0]

    first_obs = np.zeros(len(Y)).astype('int32')
    last_obs = np.zeros(len(Y)).astype('int32')

    # i iterates y
    i = 0
    # j iterates X
    j = 0
    start = 0
    end = 0
    epsilon = np.finfo(np.float32).eps
    while i < n:
        if verbose and i == 0 or i % 1000 == 999 or i == n-1:
            stderr('\r%d/%d' %(i+1, n))

        # Check if we've entered a new series in y
        if (Y_id_vectors[i] != Y_id).any():
            start = end = j
            Y_id = Y_id_vectors[i]

        # Move the X pointer forward until we are either in the same series as y or at the end of the table.
        # However, if we are already at the end of the current time series, stay put in case there are subsequent observations of the response.
        if j == 0 or (j > 0 and (X_id_vectors[j-1] != Y_id).any()):
            while j < m and (X_id_vectors[j] != Y_id).any():
                j += 1
                start = end = j

        # Move the X pointer forward until we are either at the end of the series or have moved later in time than y
        while j < m and X_time[j] <= (Y_time[i] + epsilon) and (X_id_vectors[j] == Y_id).all():
            j += 1
            end = j

        if forward:
            # Shift bounds since we're implicitly slicing backwards
            _start = X_ix[end] + 1
            _end = X_ix[start] + 1
            if np.isfinite(window_length):
                _end = min(_end, _start + window_length)
        else:
            _start = X_ix[start]
            _end = X_ix[end]
            if np.isfinite(window_length):
                _start = max(_start, _end - window_length)
        first_obs[i] = _start
        last_obs[i] = _end

        i += 1

    first_obs = first_obs[Y_ix]
    last_obs = last_obs[Y_ix]

    if exclude_instantaneous:
        _first_obs = np.where(first_obs < len(X_src), first_obs, np.maximum(0, first_obs-1))
        shift = np.fabs(X_src.time.values[_first_obs] - Y_src.time.values) <= epsilon
        if forward:
            first_obs += shift
            last_obs += shift
        else:
            first_obs -= shift
            last_obs -= shift

    stderr('\n')

    return first_obs, last_obs


def compute_filters(Y, filters=None):
    """
    Compute filters given a filter map.

    :param Y: ``pandas`` ``DataFrame``; response data.
    :param filters: ``list``; list of key-value pairs mapping column names to filtering criteria for their values.
    :return: ``numpy`` vector; boolean mask to use for ``pandas`` subsetting operations.
    """

    if filters is None:
        return Y
    select = np.ones(len(Y), dtype=bool)
    for f in filters:
        field = f[0]
        cond = f[1]
        if field in Y:
            select &= compute_filter(Y, field, cond)
        elif field.lower().endswith('nunique'):
            name = field[:-7]
            if name in Y.columns:
                vals, counts = np.unique(Y[name][select], return_counts=True)
                count_map = {}
                for v, c in zip(vals, counts):
                    count_map[v] = c
                Y[field] = Y[name].map(count_map)
                select &= compute_filter(Y, field, cond)
            else:
                stderr('Skipping unique-counts filter for column "%s", which was not found in the data...\n' % name)
        else:
            stderr('Skipping filter for column "%s", which was not found in the data...\n' % field)
    return select


def compute_filter(y, field, cond):
    """
    Compute filter given a field and condition

    :param y: ``pandas`` ``DataFrame``; response data.
    :param field: ``str``; name of column on whose values to filter.
    :param cond: ``str``; string representation of condition to use for filtering.
    :return: ``numpy`` vector; boolean mask to use for ``pandas`` subsetting operations.
    """

    assert isinstance(cond, str), 'Argument ``cond`` must be of type ``str``.'

    cond = cond.strip()
    if cond.startswith('<='):
        op = '<='
        var = cond[2:].strip()
    elif cond.startswith('>='):
        op = '<='
        var = cond[2:].strip()
    elif cond.startswith('<'):
        op = '<'
        var = cond[1:].strip()
    elif cond.startswith('>'):
        op = '>'
        var = cond[1:].strip()
    elif cond.startswith('=='):
        op = '=='
        var = cond[2:].strip()
    elif cond.startswith('!='):
        op = '!='
        var = cond[2:].strip()
    else:
        raise ValueError('Unrecognized filtering condition: %s' % cond)

    if var == 'inf':
        var = np.inf
    else:
        try:
            var = float(var)
        except ValueError:
            if var in y and not var in y[field].unique():
                var = y[var]

    if op == '<=':
        return ~pd.isna(y[field]) & (y[field] <= var)
    if cond.startswith('>='):
        return ~pd.isna(y[field]) & (y[field] >= var)
    if cond.startswith('<'):
        return ~pd.isna(y[field]) & (y[field] < var)
    if cond.startswith('>'):
        return ~pd.isna(y[field]) & (y[field] > var)
    if cond.startswith('=='):
        try:
            return ~pd.isna(y[field]) & (y[field] == var)
        except:
            return ~pd.isna(y[field]) & (y[field].astype('str') == var)
    if cond.startswith('!='):
        try:
            return ~pd.isna(y[field]) & (y[field] != var)
        except:
            return ~pd.isna(y[field]) & (y[field].astype('str') != var)
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


def expand_impulse_sequence(
        X, X_time, first_obs, last_obs, window_length, int_type='int32', float_type='float32', fill=0.):
    """
    Expand out impulse stream in **X** for each response in the target data.

    :param X: ``pandas`` ``DataFrame``; impulse (predictor) data.
    :param X_time: ``pandas`` ``Series``; timestamps associated with each impulse in **X**.
    :param first_obs: ``pandas`` ``Series``; vector of row indices in **X** of the first impulse in the time series associated with each response.
    :param last_obs: ``pandas`` ``Series``; vector of row indices in **X** of the last preceding impulse in the time series associated with each response.
    :param window_length: ``int``; number of steps in time dimension of output
    :param int_type: ``str``; name of int type.
    :param float_type: ``str``; name of float type.
    :param fill: ``float``; fill value for padding cells.
    :return: 3-tuple of ``numpy`` arrays; the expanded impulse array, the expanded timestamp array, and a boolean mask zeroing out locations of non-existent impulses.
    """

    INT_NP = getattr(np, int_type)
    FLOAT_NP = getattr(np, float_type)
    last_obs = np.array(last_obs, dtype=INT_NP)
    first_obs = np.array(first_obs, dtype=INT_NP)
    X_time = np.array(X_time, dtype=FLOAT_NP)
    X = np.array(X)

    X_2d = np.full((first_obs.shape[0], window_length, X.shape[1]), fill, dtype=FLOAT_NP)
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


def compute_time_mask(
        X_time,
        first_obs,
        last_obs,
        history_length=128,
        future_length=0,
        int_type='int32',
        float_type='float32'
):
    """
    Compute mask for expanded impulse data zeroing out non-existent impulses.

    :param X_time: ``pandas`` ``Series``; timestamps associated with each impulse in **X**.
    :param first_obs: ``pandas`` ``Series``; vector of row indices in **X** of the first impulse in the time series associated with each response.
    :param last_obs: ``pandas`` ``Series``; vector of row indices in **X** of the last preceding impulse in the time series associated with each response.
    :param history_length: ``int``; maximum number of history (backward) observations.
    :param future_length: ``int``; maximum number of future (forward) observations.
    :param int_type: ``str``; name of int type.
    :param float_type: ``str``; name of float type.
    :return: ``numpy`` array; boolean impulse mask.
    """

    INT_NP = getattr(np, int_type)
    FLOAT_NP = getattr(np, float_type)
    first_obs = np.array(first_obs, dtype=INT_NP)
    last_obs = np.array(last_obs, dtype=INT_NP)
    X_time = np.array(X_time, dtype=FLOAT_NP)

    time_mask = np.zeros((first_obs.shape[0], history_length + future_length), dtype=FLOAT_NP)

    for i, first, last in zip(np.arange(first_obs.shape[0]), first_obs, last_obs):
        sXt = X_time[first:last]
        time_mask[i][-len(sXt):] = 1

    return time_mask


def preprocess_data(
        X,
        Y,
        formula_list,
        series_ids,
        filters=None,
        history_length=128,
        future_length=0,
        all_interactions=False,
        verbose=True,
        debug=False
):
    """
    Preprocess CDR data.

    :param X: list of ``pandas`` tables; impulse (predictor) data.
    :param Y: list of ``pandas`` tables; response data.
    :param formula_list: ``list`` of ``Formula``; CDR formula for which to preprocess data.
    :param series_ids: ``list`` of ``str``; column names whose jointly unique values define unique time series.
    :param filters: ``list``; list of key-value pairs mapping column names to filtering criteria for their values.
    :param history_length: ``int``; maximum number of history (backward) observations.
    :param future_length: ``int``; maximum number of future (forward) observations.
    :param all_interactions: ``bool``; add powerset of all conformable interactions.
    :param verbose: ``bool``; whether to report progress to stderr
    :param debug: ``bool``; print debugging information
    :return: 7-tuple; predictor data, response data, filtering mask, response-aligned predictor names, response-aligned predictors, 2D predictor names, and 2D predictors
    """

    if verbose:
        stderr('Pre-processing data...\n')

    if not isinstance(X, list):
        X = [X]
    if not isinstance(Y, list):
        Y = [Y]

    select = []
    for i, _Y in enumerate(Y):
        if filters is None:
            _select = np.full((len(_Y),), True, dtype='bool')
        else:
            _select = compute_filters(_Y, filters)
            _Y = _Y[_select]
        Y[i] = _Y
        select.append(_select)

    X_response_aligned_predictor_names = None
    X_response_aligned_predictors = None
    X_2d_predictor_names = None
    X_2d_predictors = None

    if history_length or future_length:
        X_new = []
        for i in range(len(X)):
            _X = X[i]
            if verbose:
                stderr('Computing time windows for each regression target in predictor file %d...\n' % (i+1))
            for j, _Y in enumerate(Y):
                if history_length:
                    if future_length:
                        stderr('Backward...\n')
                    first_obs, last_obs = get_time_windows(
                        _X,
                        _Y,
                        series_ids,
                        window_length=history_length
                    )
                    first_obs_b, last_obs_b = first_obs, last_obs
                    exclude_instantaneous = True
                else:
                    first_obs = last_obs = None
                    exclude_instantaneous = False
                if future_length:
                    if history_length:
                        stderr('Forward...\n')
                    _first_obs, last_obs = get_time_windows(
                        _X,
                        _Y,
                        series_ids,
                        forward=True,
                        window_length=future_length,
                        exclude_instantaneous=exclude_instantaneous
                    )
                    first_obs_f, last_obs_f = _first_obs, last_obs
                    if first_obs is None:
                        first_obs = _first_obs
                _Y['first_obs_%d' % i] = first_obs
                _Y['last_obs_%d' % i] = last_obs

                if debug:
                    sample = np.random.randint(0, len(_Y), 10)
                    sample = np.concatenate([np.zeros((1,), dtype='int'), sample, np.ones((1,), dtype='int') * (len(_Y) - 1)], axis=0)

                    for k in sample:
                        print('Obs ix')
                        print(k)
                        row = _Y.iloc[k]
                        print('First ix')
                        print(first_obs[k])
                        print('Last ix')
                        print(last_obs[k])
                        print('Target:')
                        print(_Y[['subject', 'docid', 'word', 'time', 'first_obs_%d' % i, 'last_obs_%d' % i]].iloc[max(0, k-5):k+5])
                        print('Impulses:')
                        print(_X[['subject', 'docid', 'word', 'time']][row['first_obs_%d' % i]:row['last_obs_%d' % i]])
                        print('Impulses (bw):')
                        print(_X[['subject', 'docid', 'word', 'time']][first_obs_b[k]:last_obs_b[k]])
                        print('Impulses (fw):')
                        print(_X[['subject', 'docid', 'word', 'time']][first_obs_f[k]:last_obs_f[k]])
                        print()

                Y[j] = _Y

            X_new.append(_X)

        for x in formula_list:
            X_new, Y, X_response_aligned_predictor_names, X_response_aligned_predictors, X_2d_predictor_names, X_2d_predictors = x.apply_formula(
                X_new,
                Y,
                X_2d_predictor_names=X_2d_predictor_names,
                X_2d_predictors=X_2d_predictors,
                X_response_aligned_predictor_names=X_response_aligned_predictor_names,
                X_response_aligned_predictors=X_response_aligned_predictors,
                history_length=history_length,
                future_length=future_length,
                all_interactions=all_interactions,
                series_ids=series_ids
            )
    else:
        X_new = X

    return X_new, Y, select, X_response_aligned_predictor_names, X_response_aligned_predictors, X_2d_predictor_names, X_2d_predictors


def split_cdr_outputs(outputs, lengths):
    """
    Takes a dictionary of arbitrary depth containing CDR outputs with their labels as keys and splits each output into
    a list of outputs with lengths corresponding to **lengths**. Useful for aligning CDR outputs to response files,
    since multiple response files can be provided, which are underlyingly concatenated by CDR.
    Recursively modifies the dict in place.

    :param outputs: ``dict`` of arbitrary depth with ``numpy`` arrays at the leaves; the source CDR outputs
    :param lengths: array-like vector of lengths to split the outputs into
    :return: ``dict``; same key-val structure as **outputs** but with each leaf split into a list of ``len(lengths)`` vectors, one for each length value.
    """

    for k in outputs:
        if isinstance(outputs[k], dict):
            split_cdr_outputs(outputs[k], lengths)
        else:
            splits = np.cumsum(lengths)
            outputs[k] = np.split(outputs[k], splits, axis=0)

    return outputs
