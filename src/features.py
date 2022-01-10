# -*- coding: utf-8 -*-
"""
features.py
===========

Feature extractors.
"""

import re
from transformers import pipeline
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


class SemanticBaseline:

    def __str__(self):
        return 'SemanticBaseline()'

    def prepare(self, samples):
        pass

    def extract(self, sample):
        features = [
            sample.count('Agamemnon'),
            sample.count('Achilles'),
            sample.count('Atreus'),
            sample.count('Priam'),  # account for different spelling of Priam[ou]s.
            sample.count('Klytai')]  # account for different spelling of Klytaimn?estra.
        return features


class SentenceCount:

    def __str__(self):
        return 'PunctuationCount()'

    def prepare(self, samples):
        pass

    def extract(self, sample):
        return [len(re.findall(r'[a-zA-Z][.!?]', sample))]


class FrequentWords:

    def __init__(self, k, exclude_stopwords=False):
        self.k = k
        self.exclude_stopwords = exclude_stopwords

    def __str__(self):
        return f"FrequentWords(k={self.k}, exclude_stopwords={self.exclude_stopwords})"

    def prepare(self, samples):
        self.cv = CountVectorizer(max_features=self.k, stop_words=None
                                  if self.exclude_stopwords else stopwords.words('dutch'))
        self.cv.fit(samples)

    def extract(self, sample):
        return self.cv.transform([sample]).toarray()[0]


class AlphabetFrequency:

    def __str__(self):
        return 'AlphabetFrequency()'

    def prepare(self, samples):
        pass

    def extract(self, sample):
        features = []

        for i in range(ord('a'), ord('z') + 1):
            features.append(sample.lower().count(chr(i)))

        total = sum(features)
        return [f / total for f in features]


class CharacterNGrams:

    def __init__(self, n, k):
        self.n = n
        self.k = k

    def __str__(self):
        return f"CharacterNGrams(k={self.k}, n={self.n})"

    def prepare(self, samples):
        self.cv = CountVectorizer(max_features=self.k, analyzer='char', ngram_range=(self.n, self.n))
        self.cv.fit(samples)

    def extract(self, sample):
        return self.cv.transform([sample]).toarray()[0]


class POS:

    all_tags = ['ADJ', 'BW', 'LET', 'LID', 'N', 'O', 'SPEC', 'TSW', 'TW', 'VG', 'VNW', 'VZ', 'WW']

    def __init__(self, tags=[]):
        if tags:
            assert all(tag in self.all_tags for tag in tags)
            self.tags = tags
        else:
            self.tags = self.all_tags

        self.tagger = pipeline('token-classification', model='wietsedv/bert-base-dutch-cased-finetuned-lassysmall-pos')

    def __str__(self):
        return f"POS(tags=[{', '.join(self.tags)}])"

    def prepare(self, samples):
        pass

    def extract(self, sample):
        features = []

        result = self.tagger(sample)

        for tag in self.tags:
            features.append(sum(1 for token in result if tag in token['entity']))

        return features
