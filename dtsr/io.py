import sys
import pandas as pd
import numpy as np

def read_data(path_X, path_y, series_ids, categorical_columns=None):
    sys.stderr.write('Loading data...\n')
    X = pd.read_csv(path_X, sep=' ', skipinitialspace=True)
    y = pd.read_csv(path_y, sep=' ', skipinitialspace=True)

    sys.stderr.write('Ensuring sort order...\n')
    X.sort_values(series_ids + ['time'], inplace=True)
    y.sort_values(series_ids + ['time'], inplace=True)

    if categorical_columns is not None:
        for col in categorical_columns:
            X[col] = X[col].astype('category')
            y[col] = y[col].astype('category')

    X['trial'] = X.groupby(series_ids).rate.cumsum()
    X_groups = X.groupby(series_ids)
    X['percentTrialsComplete'] = X_groups['trial'].apply(lambda x: x / max(x))
    X['percentTimeComplete'] = X_groups['time'].apply(lambda x: x / max(x))
    X_groups = X.groupby(series_ids + ['sentid'])
    X['percentSentComplete'] = X_groups['sentpos'].apply(lambda x: x / max(x))
    X._get_numeric_data().fillna(value=0, inplace=True)
    return X, y