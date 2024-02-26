import sys
import pandas as pd

from .util import stderr

def read_tabular_data(X_paths, Y_paths, series_ids, categorical_columns=None, sep=' ', verbose=True):
    """
    Read impulse and response data into pandas dataframes and perform basic pre-processing.

    :param X_paths: ``str`` or ``list`` of ``str``; path(s) to impulse (predictor) data (multiple tables are concatenated). Each path may also be a ``;``-delimited list of paths to files containing predictors with different timestamps, where the predictors in each file are all timestamped with respect to the same reference point.
    :param Y_paths: ``str`` or ``list`` of ``str``; path(s) to response data (multiple tables are concatenated). Each path may also be a ``;``-delimited list of paths to files containing different response variables with different timestamps, where the response variables in each file are all timestamped with respect to the same reference point.
    :param series_ids: ``list`` of ``str``; column names whose jointly unique values define unique time series.
    :param categorical_columns: ``list`` of ``str``; column names that should be treated as categorical.
    :param sep: ``str``; string representation of field delimiter in input data.
    :param verbose: ``bool``; whether to log progress to stderr.
    :return: 2-tuple of list(``pandas`` DataFrame); (impulse data, response data). X and Y each have one element for each dataset in X_paths/Y_paths, each containing the column-wise concatenation of all column files in the path.
    """

    if not isinstance(X_paths, list):
        X_paths = [X_paths]
    if not isinstance(Y_paths, list):
        Y_paths = [Y_paths]

    if verbose:
        stderr('Loading data...\n')
    X = []
    Y = []

    for path in X_paths:
        assert path is not None, 'No data path provided. Exiting.'
        _X = []
        for x in path.split(';'):
            _X.append(pd.read_csv(x, sep=sep, skipinitialspace=True))
        X.append(_X)

    for path in Y_paths:
        assert path is not None, 'No data path provided. Exiting.'
        _Y = []
        for y in path.split(';'):
            _Y.append(pd.read_csv(y, sep=sep, skipinitialspace=True))
        Y.append(_Y)

    # Regroup by column
    
    # Stimuli
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
        
    # Responses
    Y_new = []
    # Loop through datasets
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            while j >= len(Y_new):
                Y_new.append([])
            Y_new[j].append(Y[i][j])
    Y = []
    # Loop through column files
    for x in Y_new:
        Y.append(pd.concat(x, axis=0))

    # Sort

    if verbose:
        stderr('Ensuring sort order...\n')
    for i, x in enumerate(X):
        X[i] = x.sort_values(series_ids + ['time']).reset_index(drop=True)
    for i, y in enumerate(Y):
        Y[i] = y.sort_values(series_ids + ['time']).reset_index(drop=True)

    # Process categorical

    if categorical_columns is not None:
        for t in categorical_columns:
            split = t.split(':')
            for col in split:
                for _X in X:
                    if col in _X:
                        _X[col] = _X[col].astype('category')
                for _Y in Y:
                    if col in _Y:
                        _Y[col] = _Y[col].astype('category')
            if len(split) > 1:
                for _Y in Y:
                    new_col = None
                    for col in split:
                        assert col in _Y, 'Members of categorical interaction grouping indices must all be present in every response table.'
                        if new_col is None:
                            new_col = _Y[col].astype(str)
                        else:
                            new_col = new_col + ':' + _Y[col].astype(str)
                    _Y[t] = new_col

    # Add columns to X

    for _X in X:
        assert not 'rate' in _X, '"rate" is a reserved column name in CDR. Rename your input column...'
        _X['rate'] = 1.
        if 'trial' not in _X:
            if series_ids:
                _X['trial'] = _X.groupby(series_ids).rate.cumsum()
            else:
                _X['trial'] = _X.rate.cumsum()

    return X, Y
