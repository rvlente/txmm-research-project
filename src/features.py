# -*- coding: utf-8 -*-
"""
features.py
===========

Feature extractors.
"""

from transformers import pipeline


class SentenceCount:

    def extract(self, sample):
        features = [sample.count('.')]
        return features


class AlphabetCount:

    def extract(self, sample):
        features = []

        for i in range(ord('a'), ord('z') + 1):
            features.append(sample.lower().count(chr(i)))

        return features


class POSFrequency:
    all_tags = ['ADJ', 'BW', 'LET', 'LID', 'N', 'O', 'SPEC', 'TSW', 'TW', 'VG', 'VNW', 'VZ', 'WW']

    def __init__(self, tags=[]):
        if tags:
            assert all(tag in self.all_tags for tag in tags)
            self.tags = tags
        else:
            self.tags = self.all_tags

        self.tagger = pipeline('token-classification', model='wietsedv/bert-base-dutch-cased-finetuned-lassysmall-pos')

    def extract(self, sample):
        features = []

        result = self.tagger(sample)

        for tag in self.tags:
            features.append(sum(1 for token in result if tag in token['entity']))

        return features


class WordFrequency:

    def __init__(self, k=400):
        self.k = 400

    def extract(self, sample):
        features = []
        return features
