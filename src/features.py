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
from flair.models import SequenceTagger
from flair.data import Sentence


class SemanticBaseline:

    def __init__(self, corpus):
        assert corpus in ['I', 'II']
        self.corpus = corpus

    def __str__(self):
        return f"SemanticBaseline(corpus={self.corpus})"

    def prepare(self, samples):
        pass

    def extract(self, sample):

        if self.corpus == 'I':
            features = [
                sample.count('Agamemnon'),
                sample.count('Achilles'),
                sample.count('Atreus'),
                sample.count('Priam'),  # account for different spelling of Priam[ou]s.
                sample.count('Klytai')]  # account for different spelling of Klytaimn?estra.
        else:
            features = [
                sample.count('Chichikov') + sample.count('Tchitchikov'),
                sample.count('Manilov'),
                sample.count('Nozdrev') + sample.count('Nozdryov'),
                sample.count('Plyushkin') + sample.count('Plushkin'),
                sample.count('Nikolai'),
                sample.count('Petrovitch'),
                sample.count('Kirsanov'),
                sample.count('Arkady'),
                sample.count('Sergievna') + sample.count('Sergyevna'),
                sample.count('Katia') + sample.count('Katya'),
                sample.count('Fenitchka') + sample.count('Thenichka'),
            ]

        return features


class SentenceCount:

    def __str__(self):
        return 'PunctuationCount()'

    def prepare(self, samples):
        pass

    def extract(self, sample):
        return [len(re.findall(r'[a-zA-Z][.!?]', sample))]


class FrequentWords:

    def __init__(self, k, exclude_stopwords=None):
        self.k = k

        assert exclude_stopwords in [None, 'nl', 'en']
        self.exclude_stopwords = exclude_stopwords

        if self.exclude_stopwords == 'nl':
            self.stopwords = stopwords.words('dutch')
        elif self.exclude_stopwords == 'en':
            self.stopwords = stopwords.words('english')
        else:
            self.stopwords = None

    def __str__(self):
        return f"FrequentWords(k={self.k}, exclude_stopwords={self.exclude_stopwords})"

    def prepare(self, samples):
        self.cv = CountVectorizer(max_features=self.k, stop_words=self.stopwords)
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
        return [f / total for f in features] if total > 0 else features


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

    def __init__(self, lang):
        assert lang in ['nl', 'en']
        self.lang = lang

        if self.lang == 'nl':
            self.tags = ['ADJ', 'BW', 'LET', 'LID', 'N', 'O', 'SPEC', 'TSW', 'TW', 'VG', 'VNW', 'VZ', 'WW']
            self.tagger = pipeline('token-classification',
                                   model='wietsedv/bert-base-dutch-cased-finetuned-lassysmall-pos')
        else:
            self.tags = [
                'CC', 'CD', 'DT', 'EX', 'FW', 'HYPH', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NFP', 'NN', 'NNP', 'NNPS',
                'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG',
                'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
            self.tagger = SequenceTagger.load('flair/pos-english')

    def __str__(self):
        return f"POS(lang={self.lang}, tags=[{', '.join(self.tags)}])"

    def prepare(self, samples):
        pass

    def extract(self, sample):
        features = []

        if self.lang == 'nl':
            tokens = [token['entity'] for token in self.tagger(sample)]
        else:
            sentence = Sentence(sample)
            self.tagger.predict(sentence)
            tokens = [token.tag for token in sentence.get_spans()]

        for tag in self.tags:
            features.append(sum(1 for token in tokens if tag in token))

        return features
