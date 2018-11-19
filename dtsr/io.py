import sys
import pandas as pd

def read_data(path_X, path_y, series_ids, categorical_columns=None, sep=' '):
    """
    Read impulse and response data into pandas dataframes and perform basic pre-processing.

    :param path_X: ``str`` or ``list`` of ``str``; path(s) to impulse (predictor) data (multiple tables are concatenated).
    :param path_y: ``str`` or ``list`` of ``str``; path(s) to response data (multiple tables are concatenated).
    :param series_ids: ``list`` of ``str``; column names whose jointly unique values define unique time series.
    :param categorical_columns: ``list`` of ``str``; column names that should be treated as categorical.
    :param sep: ``str``; string representation of field delimiter in input data.
    :return: 2-tuple of ``pandas`` ``DataFrame``; (impulse data, response data)
    """

    if not isinstance(path_X, list):
        path_X = [path_X]
    if not isinstance(path_y, list):
        path_y = [path_y]

    sys.stderr.write('Loading data...\n')
    X = []
    y = []

    for path in path_X:
        X.append(pd.read_csv(path, sep=sep, skipinitialspace=True))

    for path in path_y:
        y.append(pd.read_csv(path, sep=sep, skipinitialspace=True))

    X = pd.concat(X, axis=0)
    y = pd.concat(y, axis=0)

    sys.stderr.write('Ensuring sort order...\n')
    X.sort_values(series_ids + ['time'], inplace=True)
    y.sort_values(series_ids + ['time'], inplace=True)

    if categorical_columns is not None:
        for col in categorical_columns:
            if col in X.columns:
                X[col] = X[col].astype('category')
            if col in y.columns:
                y[col] = y[col].astype('category')

    X['rate'] = 1.
    X['trial'] = X.groupby(series_ids).rate.cumsum()
    # X_groups = X.groupby(series_ids)
    # X['percentTrialsComplete'] = X_groups['trial'].apply(lambda x: x / max(x))
    # X['percentTimeComplete'] = X_groups['time'].apply(lambda x: x / max(x))
    # X_groups = X.groupby(series_ids + ['sentid'])
    # X['percentSentComplete'] = X_groups['sentpos'].apply(lambda x: x / max(x))
    # X._get_numeric_data().fillna(value=0, inplace=True)
    return X, y
