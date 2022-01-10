# -*- coding: utf-8 -*-
"""
data.py
=======

Utility class for loading the dataset and generating samples.
"""

from pathlib import Path
import matplotlib.pyplot as plt


class TranslationMatrixCorpus:

    def __init__(self, folder, authors, translators):
        self.folder = Path(folder)
        self.authors = authors
        self.translators = translators

        assert self._verify_data(), f'Data invalid for {authors}, {translators}.'

        self.data = {author: {translator: self._load_author(author, translator) for translator in self.translators}
                     for author in self.authors}

        self.raw = [self.data[author][translator] for author in self.authors for translator in self.translators]

    def _verify_data(self):
        for author in self.authors:
            for translator in self.translators:
                if not (self.folder / author / f'{translator}.txt').exists():
                    return False
        return True

    def _load_author(self, author, translator):
        with open(self.folder / author / f'{translator}.txt') as f:
            return f.read()

    @staticmethod
    def _count_words(line):
        return len(line.split(' '))

    def _yield_samples(self, author, translator, min_words):
        author_data = self.data[author][translator]
        lines = [line for line in author_data.split('\n') if line]
        sample = ''
        wc = 0

        for line in lines:
            sample += line
            sample += '\n'
            wc += self._count_words(line)

            if wc > min_words:
                yield sample
                sample = ''
                wc = 0

        yield sample

    def create_samples(self, min_words=500):
        samples = []

        for author in self.authors:
            for translator in self.translators:
                samples.extend((sample, author, translator)
                               for sample in self._yield_samples(author, translator, min_words))

        return [s[0] for s in samples], [s[1] for s in samples], [s[2] for s in samples]

    def plot_distribution(self, min_words=500):
        X, y_author, y_translator = self.create_samples(min_words)

        size = len(X)
        author_labels = sorted(hash(x) for x in y_author)
        translator_labels = sorted(hash(x) for x in y_translator)

        plt.scatter(range(size), [0] * size, c=author_labels, marker="_", lw=20)
        plt.scatter(range(size), [1] * size, c=translator_labels, marker="_", lw=20)

        plt.xlim([0, size - 1])
        plt.ylim([-0.6, 1.6])
        plt.yticks([0, 1], ['author', 'translator'])
        plt.xlabel('Sample index')

        plt.show()
