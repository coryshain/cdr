import sys
import os
import argparse
import pandas as pd
import numpy as np


# Thanks to SO user gbtimmon (https://stackoverflow.com/users/1158990/gbtimmon) for this wrapper to prevent modules from printing
def suppress_output(func):
    def wrapper(*args, **kwargs):
        with open(os.devnull,"w") as devNull:
            original = sys.stdout
            sys.stdout = devNull
            out = func(*args, **kwargs)
            sys.stdout = original
            return out
    return wrapper


def compute_history_intervals(X, y, series_ids):
    m = len(X)
    n = len(y)

    time_X = np.array(X.time)
    time_y = np.array(y.time)
    series_starts_X = np.zeros_like(time_X, dtype=bool)
    series_starts_X[0] = True
    series_starts_y = np.zeros_like(time_y, dtype=bool)
    series_starts_y[0] = True

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
        if i == 0 or i % 1000 == 999  or i == len(y) - 1:
            sys.stderr.write('\r%d/%d' % (i+1, len(y)))
            sys.stderr.flush()
        # Check if we've entered a new series in y
        if (id_vectors_y[i] != y_cur_ids).any():
            series_starts_y[i] = True
            start = end = j
            y_cur_ids = id_vectors_y[i]

        # Move the X pointer forward until we are either in the same series as y or at the end of the table.
        # However, if we are already at the end of the current time series, stay put in case there are subsequent observations of the response.
        if j == 0 or (j > 0 and (id_vectors_X[j-1] != y_cur_ids).any()):
            while j < m and (id_vectors_X[j] != y_cur_ids).any():
                if j > 0 and (id_vectors_X[j] != id_vectors_X[j-1]).any():
                    series_starts_X[j] = True
                    X_cur_ids = id_vectors_X[j]
                j += 1
                start = end = j

        # Move the X pointer forward until we are either at the end of the series or have moved later in time than y
        while j < m and time_X[j] <= (time_y[i] + epsilon) and (id_vectors_X[j] == y_cur_ids).all():
            if j > 0 and (id_vectors_X[j] != id_vectors_X[j - 1]).any():
                series_starts_X[j] = True
                X_cur_ids = id_vectors_X[j]
            j += 1
            end = j

        first_obs[i] = start
        last_obs[i] = end

        i += 1

    sys.stderr.write('\n')

    return first_obs, last_obs, series_starts_X, series_starts_y


def convolve(X, y, first_obs, last_obs, columns):
    time_X = np.array(X.time)
    time_y = np.array(y.time)

    X = np.array(X[columns])
    X_conv = np.zeros((y.shape[0], X.shape[1]))

    for i in range(len(y)):
        if i == 0 or i % 1000 == 999  or i == len(y) - 1:
            sys.stderr.write('\r%d/%d' % (i+1, len(y)))
            sys.stderr.flush()
        s, e = first_obs[i], last_obs[i]
        hrf_weights = hrf(time_y[i] - time_X[s:e])[..., None]
        X_conv[i] = (X[s:e] * hrf_weights).sum(axis=0)

    X_conv = pd.DataFrame(X_conv, columns=columns, index=y.index)
    y = pd.concat([y, X_conv], axis=1)

    sys.stderr.write('\n')

    return y

def interpolate(X, y, first_obs, last_obs, series_starts_X, series_starts_y, columns, n_filters=4, mode='linear'):
    time_X = np.array(X.time)
    time_y = np.array(y.time)

    X = np.array(X[columns])
    X_interp = [np.zeros((y.shape[0], X.shape[1])) for i in range(n_filters)]

    for i in range(len(y)):
        if i == 0 or i % 1000 == 999  or i == len(y) - 1:
            sys.stderr.write('\r%d/%d' % (i+1, len(y)))
            sys.stderr.flush()

        s, e = first_obs[i], last_obs[i]
        if s > 0 and not series_starts_X[s]:
            s = - 1
        if e >= len(series_starts_X) - 1 or not series_starts_X[e]:
            e += 1
        if e-s > 0:
            if mode.lower() == 'linear':
                f = interp1d(time_X[max(0,s-1):e+1], X[max(0,s-1):e+1], axis=0, bounds_error=False, fill_value=0.)
                for j in range(n_filters):
                    for j in range(n_filters):
                        if j > 0 and (i+j > len(y) - 1 or series_starts_y[i+j]):
                            break
                        else:
                            X_interp[j][i] = f(time_y[i - j])
            else:
                raise ValueError('Unrecognized interpolation mode "%s"' % mode)

    X_interp = np.concatenate(X_interp, axis=1)

    new_cols = []
    for i in range(n_filters):
        for c in columns:
            new_cols.append(c + 'S%d' % i)

    X_interp = pd.DataFrame(X_interp, columns=new_cols, index=y.index)
    y = pd.concat([y, X_interp], axis=1)

    sys.stderr.write('\n')

    return y


def lanczos(X, y, first_obs, last_obs, series_starts_X, series_starts_y, columns, n_filters=4):
    # Assumes X and y contain the same time series in the same order
    n = len(y)
    time_X = np.split(np.array(X.time), np.where(series_starts_X)[0], axis=0)
    if len(time_X) > 0 and len(time_X[0]) == 0:
        time_X = time_X[1:]
    time_y = np.split(np.array(y.time), np.where(series_starts_y)[0], axis=0)
    if len(time_y) > 0 and len(time_y[0]) == 0:
        time_y = time_y[1:]
    X = np.split(np.array(X[columns]), np.where(series_starts_X)[0], axis=0)
    if len(X) > 0 and len(X[0]) == 0:
        X = X[1:]
    i = 0 # iterates y
    j = 0 # iterates X

    X_lanczos = [[] for _ in range(n_filters)]
    for i in range(len(time_y)):
        if i == 0 or i % 10 == 9 or i == len(time_y) - 1:
            sys.stderr.write('\r%d/%d' % (i+1, len(time_y)))
            sys.stderr.flush()

        X_cur = lanczosinterp2D(X[i], time_X[i], time_y[i])
        X_lanczos[0].append(X_cur)
        for j in range(1, n_filters):
            X_shift = np.zeros_like(X_cur)
            X_shift[j:] = X_cur[:-j]
            X_lanczos[j].append(X_shift)
    X_lanczos = [np.concatenate(x, axis=0) for x in X_lanczos]
    X_lanczos = np.concatenate(X_lanczos, axis=1)

    new_cols = []
    for i in range(n_filters):
        for c in columns:
            new_cols.append(c + 'S%d' % i)

    X_lanczos = pd.DataFrame(X_lanczos, columns=new_cols, index=y.index)
    y = pd.concat([y, X_lanczos], axis=1)

    sys.stderr.write('\n')

    return y


def aggregate(X, y, first_obs, last_obs, series_starts_y, columns, n_filters=4, mode='mean'):
    time_X = np.array(X.time)
    time_y = np.array(y.time)
    time_y_prev = np.pad(time_y[:-1], (1,0), mode='constant')
    time_y_prev = np.where(series_starts_y, np.zeros_like(time_y), time_y_prev)

    X = np.array(X[columns])
    X_aggregate = [np.zeros((y.shape[0], X.shape[1])) for _ in range(n_filters)]

    for i in range(len(y)):
        if i == 0 or i % 1000 == 999  or i == len(y) - 1:
            sys.stderr.write('\r%d/%d' % (i+1, len(y)))
            sys.stderr.flush()

        s, e = first_obs[i], last_obs[i]

        if e-s > 0:
            mask = np.logical_and(time_X[s:e] >= time_y_prev[i], time_X[s:e] <= time_y[i])
            X_window = X[s:e][mask]
            if len(X_window) > 0:
                X_aggregate_cur = getattr(X_window, mode.lower())(axis=0)
                for j in range(n_filters):
                    if j > 0 and (i+j > len(y) - 1 or series_starts_y[i+j]):
                        break
                    else:
                        X_aggregate[j][i+j] = X_aggregate_cur

    X_aggregate = np.concatenate(X_aggregate, axis=1)

    new_cols = []
    for i in range(n_filters):
        for c in columns:
            new_cols.append(c + 'S%d' % i)

    X_aggregate = pd.DataFrame(X_aggregate, columns=new_cols, index=y.index)
    y = pd.concat([y, X_aggregate], axis=1)

    sys.stderr.write('\n')

    return y


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Align properties of naturalistic, variably-spaced stimuli to fMRI scan times, using convolution, interpolation, or averaging.
    Convolution uses the canonical HRF defined by pymvpa2.
    Interpolation uses linear interpolation + resampling.
    Averaging averages values within each fMRI TR interval.
    ''')
    argparser.add_argument('preds', nargs='+', help='Path(s) to predictor data table(s)')
    argparser.add_argument('response', type=str, help='Path to response data table')
    argparser.add_argument('-c', '--conv_cols', nargs='*', default=[], help='Columns from predictor table to convolve.')
    argparser.add_argument('-i', '--interp_cols', nargs='*', default=[], help='Columns from predictor table to interpolate and resample.')
    argparser.add_argument('-a', '--average_cols', nargs='*', default=[], help='Columns from predictor table to average within TRs.')
    argparser.add_argument('-s', '--sum_cols', nargs='*', default=[], help='Columns from predictor table to sum within TRs.')
    argparser.add_argument('-l', '--lanczos_cols', nargs='*', default=[], help='Columns from predictor table to aggregate using a Lanczos filter.')
    argparser.add_argument('-k', '--keys', nargs='+', default=['subject', 'docid', 'fROI'], help='Column names to use as keys to define time series.')
    argparser.add_argument('-f', '--n_filters', type=int, default=4, help="Number of FIR filters to output. Ignored if mode is 'convolution'.")
    args, unknown = argparser.parse_known_args()

    if len(args.conv_cols) > 0:
        from mvpa2.misc.data_generators import double_gamma_hrf as hrf
    if len(args.interp_cols) > 0:
        from scipy.interpolate import interp1d
    if len(args.lanczos_cols) > 0:
        sys.path.append('speechmodeltutorial/')
        from interpdata import lanczosinterp2D
        lanczosinterp2D = suppress_output(lanczosinterp2D)

    response = pd.read_csv(args.response,sep=' ',skipinitialspace=True)
    response.sort_values(args.keys + ['time'], inplace=True)

    conv_cols_to_process = args.conv_cols[:]
    interp_cols_to_process = args.interp_cols[:]
    average_cols_to_process = args.average_cols[:]
    sum_cols_to_process = args.sum_cols[:]
    lanczos_cols_to_process = args.lanczos_cols[:]

    for preds in args.preds:
        if len(conv_cols_to_process) > 0 or len(interp_cols_to_process) > 0 or len(average_cols_to_process) > 0 or len(sum_cols_to_process) > 0 or len(lanczos_cols_to_process) > 0:
            preds_name = preds
            preds = pd.read_csv(preds,sep=' ',skipinitialspace=True)
            preds = preds[args.keys + ['time'] + [c for c in (conv_cols_to_process + interp_cols_to_process + average_cols_to_process + sum_cols_to_process + lanczos_cols_to_process) if c in preds.columns]]
            preds.sort_values(args.keys + ['time'], inplace=True)
            if 'rate' not in preds.columns:
                preds['rate'] = 1

            sys.stderr.write('Computing series bounds for predictor file %s...\n' % preds_name)
            first_obs, last_obs, series_starts_X, series_starts_y = compute_history_intervals(preds, response, args.keys)

            conv_cols = []
            conv_cols_to_process_new = []
            for c in conv_cols_to_process:
                if c in preds.columns:
                    conv_cols.append(c)
                else:
                    conv_cols_to_process_new.append(c)
            conv_cols_to_process = conv_cols_to_process_new
            if len(conv_cols) > 0:
                sys.stderr.write('Convolving columns %s with the canonical HRF...\n' % ', '.join(conv_cols))
                response = convolve(preds, response, first_obs, last_obs, conv_cols)

            interp_cols = []
            interp_cols_to_process_new = []
            for c in interp_cols_to_process:
                if c in preds.columns:
                    interp_cols.append(c)
                else:
                    interp_cols_to_process_new.append(c)
            interp_cols_to_process = interp_cols_to_process_new
            if len(interp_cols) > 0:
                sys.stderr.write('Interpolating and resampling columns %s...\n' % ', '.join(interp_cols))
                response = interpolate(preds, response, first_obs, last_obs, series_starts_X, series_starts_y, interp_cols, n_filters=args.n_filters)

            average_cols = []
            average_cols_to_process_new = []
            for c in average_cols_to_process:
                if c in preds.columns:
                    average_cols.append(c)
                else:
                    average_cols_to_process_new.append(c)
            average_cols_to_process = average_cols_to_process_new
            if len(average_cols) > 0:
                sys.stderr.write('Averaging columns %s within TRs...\n' % ', '.join(average_cols))
                response = aggregate(preds, response, first_obs, last_obs, series_starts_y, average_cols, n_filters=args.n_filters, mode='mean')

            sum_cols = []
            sum_cols_to_process_new = []
            for c in sum_cols_to_process:
                if c in preds.columns:
                    sum_cols.append(c)
                else:
                    sum_cols_to_process_new.append(c)
            sum_cols_to_process = sum_cols_to_process_new
            if len(sum_cols) > 0:
                sys.stderr.write('Summing columns %s within TRs...\n' % ', '.join(sum_cols))
                response = aggregate(preds, response, first_obs, last_obs, series_starts_y, sum_cols, n_filters=args.n_filters, mode='sum')

            lanczos_cols = []
            lanczos_cols_to_process_new = []
            for c in lanczos_cols_to_process:
                if c in preds.columns:
                    lanczos_cols.append(c)
                else:
                    lanczos_cols_to_process_new.append(c)
            lanczos_cols_to_process = lanczos_cols_to_process_new
            if len(lanczos_cols) > 0:
                sys.stderr.write('Lanczos filtering columns %s within TRs...\n' % ', '.join(lanczos_cols))
                response = lanczos(preds, response, first_obs, last_obs, series_starts_X, series_starts_y, lanczos_cols, n_filters=args.n_filters)

    response.to_csv(sys.stdout, ' ', index=False, na_rep='NaN')

