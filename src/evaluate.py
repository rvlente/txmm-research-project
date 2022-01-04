# -*- coding: utf-8 -*-
"""
data.py
=======

Feature evaluation functions
"""

from statistics import mean

from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from tqdm import tqdm


def _cprint(msg, condition):
    """
    One-liner for conditional printing.
    """
    if condition:
        print(msg)


def evaluate(ttm, label_on='text', feature_extractors=[], report=False):
    """
    Perform a cross validation of the features on the dataset and report the evaluation measures. 
    """

    assert label_on in ['text', 'translator']
    labels = ttm.texts if label_on == 'text' else ttm.translators

    X, y = ttm.create_samples(label_on=label_on)

    # Extract features from samples.
    def _extract_features(sample):
        features = []

        for ft_ex in feature_extractors:
            features.extend(ft_ex.extract(sample))

        return features

    features = map(_extract_features, X)

    _cprint('Extracting features...', report)
    features = list(tqdm(features, total=len(X)) if report else features)

    # Cross validate classifiers.
    pipe = Pipeline([('scale', MinMaxScaler()), ('clf', LinearSVC())])
    cv = RepeatedStratifiedKFold(random_state=1337)
    scoring = {
        'precision': make_scorer(precision_score, labels=labels, average='macro'),
        'recall': make_scorer(recall_score, labels=labels, average='macro'),
        'f1': make_scorer(f1_score, labels=labels, average='macro'),
    }

    _cprint('Performing cross validation...', report)

    results = cross_validate(pipe, features, y, scoring=scoring, cv=cv)

    _cprint('Done!', report)

    return {
        'precision': mean(results['test_precision']),
        'recall': mean(results['test_recall']),
        'f1': mean(results['test_f1']),
    }
