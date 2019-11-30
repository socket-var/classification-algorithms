import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


def read_csv(file_name, remove_header=None):
    if remove_header:
        data = pd.read_csv(file_name)
    else:
        data = pd.read_csv(file_name, header=None)

    return np.array(data.iloc[:, 1:].copy())


def kfold_stratified_split(X, y, k):

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    splits = []

    for train, test in skf.split(X, y):
        new_X = X[train, :].copy()
        new_y = y[train, :].copy()

        new_X_test = X[test, :].copy()
        new_y_test = y[test, :].copy()
        splits.append([(new_X, new_y), (new_X_test, new_y_test)])

    return splits
