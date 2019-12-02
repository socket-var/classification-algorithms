import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import jaccard_similarity_score, adjusted_rand_score


def import_txt(filename):
    data = pd.read_csv(filename, sep="\t", header=None)

    data = data.sample(frac=1, random_state=42)

    X = data.iloc[:, :-1].copy()
    y = data.iloc[:, -1].copy()

    unique_labels = list(set(y))

    return np.array(X), np.array(y), unique_labels


def gini(rows):

    unique, counts = np.unique(np.array(rows)[:, -1], return_counts=True)
    counts = dict(zip(unique, counts))

    impurity = 1
    for label in counts:
        impurity -= (counts[label] / float(len(rows)))**2
    return impurity


def info_gain(true_subset, false_subset, parent_gini):
    p = len(true_subset) / float(len(true_subset) + len(false_subset))
    return parent_gini - p * gini(true_subset) - (1 - p) * gini(false_subset)


def metric_computation(validation_labels, predicted_labels):
    tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
    for i in range(len(validation_labels)):
        if(validation_labels[i] == 1.0 and predicted_labels[i] == 1.0):
            tp += 1
        elif(validation_labels[i] == 0.0 and predicted_labels[i] == 1.0):
            fp += 1
        elif(validation_labels[i] == 0.0 and predicted_labels[i] == 0.0):
            tn += 1
        else:
            fn += 1

    accuracy = (tp+tn)/(tp+tn+fp+fn)
    try:
        precision = tp/(tp+fp)
    except ZeroDivisionError:
        precision = 0

    try:
        recall = tp/(tp+fn)
    except ZeroDivisionError:
        recall = 0

    try:
        fmeasure = (2*recall*precision)/(recall+precision)
    except ZeroDivisionError:
        fmeasure = 0

    return accuracy, precision, recall, fmeasure


def cross_validation_split(k, X, y):

    n = X.shape[0] // k
    rem = X.shape[0] % k

    folds = []

    start = 0

    for i in range(k):
        if i < rem:
            end = start+n+1
        else:
            end = start+n
        fold = [X[start:end], y[start:end]]
        folds.append(fold)
        start = end

    return folds
