# -*- coding: utf-8 -*-
"""
data.py
=======

Feature evaluation functions.
"""

from statistics import mean
from sklearn import feature_extraction

from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from tqdm import tqdm


def evaluate(ttm, feature_extractors=[], report=False):
    """
    Perform a cross validation of the features on the dataset and report the evaluation measures. 
    """

    X, y_author, y_translators = ttm.create_samples()

    # Extract features from samples.
    for ft_ex in feature_extractors:
        ft_ex.prepare(X)

    def _extract_features(sample):
        features = []

        for ft_ex in feature_extractors:
            features.extend(ft_ex.extract(sample))

        return features

    features = map(_extract_features, X)
    features = list(tqdm(features, total=len(X)) if report else features)

    # Cross validate classifiers.
    pipe = Pipeline([('scale', MinMaxScaler()), ('clf', LinearSVC())])
    cv = RepeatedStratifiedKFold(random_state=1337)

    results = {}

    for label, y in zip(['author', 'translator'], [y_author, y_translators]):
        labels = list(set(y))

        scoring = {
            'precision': make_scorer(precision_score, labels=labels, average='macro', zero_division=0),
            'recall': make_scorer(recall_score, labels=labels, average='macro', zero_division=0),
            'f1': make_scorer(f1_score, labels=labels, average='macro', zero_division=0),
        }

        result = cross_validate(pipe, features, y, scoring=scoring, cv=cv, return_estimator=True)

        results[label] = {
            'precision': round(mean(result['test_precision']), 3),
            'recall': round(mean(result['test_recall']), 3),
            'f1': round(mean(result['test_f1']), 3),
        }

    return results
