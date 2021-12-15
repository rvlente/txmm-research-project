# -*- coding: utf-8 -*-
"""
features.py
===========

Feature extractors.
"""

from transformers import pipeline

tagger = pipeline('token-classification', model='wietsedv/bert-base-dutch-cased-finetuned-lassysmall-pos')


def alphabet_frequency(sample):
    features = []

    for i in range(ord('a'), ord('z') + 1):
        features.append(sample.lower().count(chr(i)))

    return features


def pos_frequency(sample):
    features = []

    result = tagger(sample)

    for tag in ['ADJ', 'BW', 'LET', 'LID', 'N', 'O', 'SPEC', 'TSW', 'TW', 'VG', 'VNW', 'VZ', 'WW']:
        features.append(sum(1 for token in result if tag in token['entity']))

    return features
