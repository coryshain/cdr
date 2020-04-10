import sys
import pandas as pd

from .util import stderr

def read_data(X_paths, y_paths, series_ids, categorical_columns=None, sep=' '):
    """
    Read impulse and response data into pandas dataframes and perform basic pre-processing.

    :param X_paths: ``str`` or ``list`` of ``str``; path(s) to impulse (predictor) data (multiple tables are concatenated). Each path may also be a ``;``-delimited list of paths to files containing predictors with different timestamps, where the predictors in each file all share the same set of timestamps.
    :param y_paths: ``str`` or ``list`` of ``str``; path(s) to response data (multiple tables are concatenated).
    :param series_ids: ``list`` of ``str``; column names whose jointly unique values define unique time series.
    :param categorical_columns: ``list`` of ``str``; column names that should be treated as categorical.
    :param sep: ``str``; string representation of field delimiter in input data.
    :return: (list(``pandas`` DataFrame), ``pandas`` DataFrame); (impulse data, response data). Impulse data has one element for each dataset in X_paths, each containing the column-wise concatenation of all column files in the path.
    """

    if not isinstance(X_paths, list):
        X_paths = [X_paths]
    if not isinstance(y_paths, list):
        y_paths = [y_paths]

    stderr('Loading data...\n')
    X = []
    y = []

    for path in X_paths:
        x_paths = []
        for x in path.split(';'):
            x_paths.append(pd.read_csv(x, sep=sep, skipinitialspace=True))
        X.append(x_paths)

    for path in y_paths:
        y.append(pd.read_csv(path, sep=sep, skipinitialspace=True))

    X_new = []
    # Loop through datasets
    for i in range(len(X)):
        for j in range(len(X[i])):
            while j >= len(X_new):
                X_new.append([])
            X_new[j].append(X[i][j])
    X = []
    # Loop through column files
    for x in X_new:
        X.append(pd.concat(x, axis=0))
    y = pd.concat(y, axis=0)

    stderr('Ensuring sort order...\n')
    for i, x in enumerate(X):
        X[i] = x.sort_values(series_ids + ['time']).reset_index(drop=True)
    y = y.sort_values(series_ids + ['time']).reset_index(drop=True)

    if categorical_columns is not None:
        for col in categorical_columns:
            for x in X:
                if col in x.columns:
                    x[col] = x[col].astype('category')
            if col in y.columns:
                y[col] = y[col].astype('category')

    for x in X:
        assert not 'rate' in x.columns, '"rate" is a reserved column name in CDR. Rename your input column...'
        x['rate'] = 1.
        if 'trial' not in x.columns:
            x['trial'] = x.groupby(series_ids).rate.cumsum()
    # X_groups = X.groupby(series_ids)
    # X['percentTrialsComplete'] = X_groups['trial'].apply(lambda x: x / max(x))
    # X['percentTimeComplete'] = X_groups['time'].apply(lambda x: x / max(x))
    # X_groups = X.groupby(series_ids + ['sentid'])
    # X['percentSentComplete'] = X_groups['sentpos'].apply(lambda x: x / max(x))
    # X._get_numeric_data().fillna(value=0, inplace=True)
    return X, y
