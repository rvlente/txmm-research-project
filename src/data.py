# -*- coding: utf-8 -*-
"""
data.py
=======

Utility class for loading the dataset and generating samples.
"""

from pathlib import Path


class TranslationMatrixCorpus:

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

    def _yield_samples(self, text, translator, min_words):
        text_data = self.data[text][translator]
        lines = [line for line in text_data.split('\n') if line]
        sample = ''
        wc = 0

        for line in lines:
            sample += line
            sample += '\n'
            wc += self.count_words(line)

            if wc > min_words:
                yield sample
                sample = ''
                wc = 0

        yield sample

    def create_samples(self, label_on='text', min_words=500):
        assert label_on in ['text', 'translator']

        samples = []

        for text in self.texts:
            for translator in self.translators:
                samples.extend((sample, text, translator)
                               for sample in self._yield_samples(text, translator, min_words))

        label_index = 1 if label_on == 'text' else 2

        X = [s[0] for s in samples]
        y = [s[label_index] for s in samples]

        return X, y
