"""
Import methods for multiple myeloma multiclass classifier
Gregory Way 2018

Usage:

    from utils import shuffle_columns
    from utils import apply_classifier
"""

import numpy as np
import pandas as pd
from sklearn.utils.extmath import safe_sparse_dot


def apply_classifier(x, w, b, proba=True, dropna=True):
    """
    Apply the classifier to additional data.
    See https://github.com/scikit-learn/scikit-learn/blob/ac1b04875331fc291a437025a18ddfefd4051d7c/sklearn/linear_model/base.py
    for more details

    Arguments:
    x - pandas dataframe: the new data trying to predict (gene by sample)
    w - pandas dataframe: the classifier coefficients (gene weight by class)
    b - pandas dataframe: the multiclass classifier intercepts (b by class)
    proba - output probability or decision scores
    dropna - decision to drop missing values. If not dropped, fill with zero.

    Import only
    """
    # Align matrices
    # NOTE: This will drop coefficients not present in X

    if dropna:
        x = x.reindex(w.index, axis='columns').dropna(axis='columns')
        w = w.reindex(x.columns).dropna()
    else:
        x = x.reindex(w.index, axis='columns', fill_value=0)
        w = w.reindex(x.columns, fill_value=0)

    scores = safe_sparse_dot(x, w, dense_output=True) + np.array(b)

    if proba:
        scores *= -1
        np.exp(scores, scores)
        scores += 1
        np.reciprocal(scores, scores)
        scores /= scores.sum(axis=1).reshape((scores.shape[0], -1))

    scores = pd.DataFrame(scores, index=x.index, columns=b.columns)

    return scores


def shuffle_columns(gene):
    """
    To be used in an `apply` pandas func to shuffle columns around a datafame
    Import only
    """
    import numpy as np
    return np.random.permutation(gene.tolist())
