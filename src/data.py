# -*- coding: utf-8 -*-
"""
data.py
=======

Utility classes for loading the dataset and generating train and test data.
"""

from pathlib import Path
from statistics import mean

from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline


class TextTranslationMatrix:

    def __init__(self, folder, texts, translators):
        self.folder = Path(folder)
        self.texts = texts
        self.translators = translators

        assert self._verify_data(), f'Data invalid for {texts}, {translators}.'

        self.data = {text: {translator: self._load_text(text, translator) for translator in self.translators}
                     for text in self.texts}

    def _verify_data(self):
        for text in self.texts:
            for translator in self.translators:
                if not (self.folder / text / f'{translator}.txt').exists():
                    return False
        return True

    def _load_text(self, text, translator):
        with open(self.folder / text / f'{translator}.txt') as f:
            return f.read()

    def count_words(self, line):
        return len(line.split(' '))

    def _yield_samples(self, text, translator):
        text_data = self.data[text][translator]
        lines = [line for line in text_data.split('\n') if line]
        sample = ''
        wc = 0

        for line in lines:
            sample += line
            sample += '\n'
            wc += self.count_words(line)

            if wc > 500:
                yield sample
                sample = ''
                wc = 0

        yield sample

    def _create_samples(self, label_on='text'):
        assert label_on in ['text', 'translator']

        samples = []

        for text in self.texts:
            for translator in self.translators:
                samples.extend((sample, text, translator) for sample in self._yield_samples(text, translator))

        label_index = 1 if label_on == 'text' else 2

        X = [s[0] for s in samples]
        y = [s[label_index] for s in samples]

        return X, y

    def cross_validate(self, label_on='text', feature_extractors=[]):
        assert label_on in ['text', 'translator']
        labels = self.texts if label_on == 'text' else self.translators

        X, y = self._create_samples(label_on=label_on)

        def _extract_features(sample):
            features = []

            for ft_ex in feature_extractors:
                features.extend(ft_ex.extract(sample))

            return features

        features = list(map(_extract_features, X))

        pipe = Pipeline([('scale', MinMaxScaler()), ('clf', LinearSVC())])
        cv = RepeatedStratifiedKFold(random_state=1337)
        scoring = {
            'precision': make_scorer(precision_score, labels=labels, average='macro'),
            'recall': make_scorer(recall_score, labels=labels, average='macro'),
            'f1': make_scorer(f1_score, labels=labels, average='macro'),
        }

        results = cross_validate(pipe, features, y, scoring=scoring, cv=cv)

        return {
            'precision': mean(results['test_precision']),
            'recall': mean(results['test_recall']),
            'f1': mean(results['test_f1']),
        }
