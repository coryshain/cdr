import sys
import pandas as pd

def read_data(path_X, path_y, series_ids, categorical_columns=None, sep=' '):
    """
    Read impulse and response data into pandas dataframes and perform basic pre-processing.

    :param path_X: ``str``; path to impulse (predictor) data.
    :param path_y: ``str``; path to response data.
    :param series_ids: ``list`` of ``str``; column names whose jointly unique values define unique time series.
    :param categorical_columns: ``list`` of ``str``; column names that should be treated as categorical.
    :param sep: ``str``; string representation of field delimiter in input data.
    :return: 2-tuple of ``pandas`` ``DataFrame``; (impulse data, response data)
    """

    sys.stderr.write('Loading data...\n')
    X = pd.read_csv(path_X, sep=sep, skipinitialspace=True)
    y = pd.read_csv(path_y, sep=sep, skipinitialspace=True)

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
