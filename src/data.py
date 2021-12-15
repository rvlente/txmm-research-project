# -*- coding: utf-8 -*-
"""
data.py
=======

Utility classes for loading the dataset and generating train and test data.
"""

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold


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

    def create_samples(self, label_on='text', split=0.5):
        assert label_on in ['text', 'translator']

        samples = []

        for text in self.texts:
            for translator in self.translators:
                samples.extend((sample, text, translator) for sample in self._yield_samples(text, translator))

        label_index = 1 if label_on == 'text' else 2

        a = [s[0] for s in samples]
        b = [s[label_index] for s in samples]

        return train_test_split(a, b, train_size=split, shuffle=True)

    def create_model(self, label_on='text', feature_extractors=[]):
        assert label_on in ['text', 'translator']
        X_train, X_test, y_train, y_test = self._create_samples(label_on=label_on)

        def extract_features(sample):
            features = []

            for f_ex in feature_extractors:
                features.extend(f_ex(sample))

            return features

        X_train = list(map(extract_features, X_train))
        X_test = list(map(extract_features, X_test))

        clf = SVC(kernel='linear')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        precision, recall, f1 = self.evaluate(y_test, y_pred, labels=self.texts
                                              if label_on == 'text' else self.translators)
        return clf, {'precision': precision, 'recall': recall, 'f1': f1}

    @staticmethod
    def evaluate(y, y_pred, labels):
        precision = precision_score(y, y_pred, labels=labels, average='macro').item()
        recall = recall_score(y, y_pred, labels=labels, average='macro').item()
        f1 = f1_score(y, y_pred, labels=labels, average='macro').item()

        return precision, recall, f1
