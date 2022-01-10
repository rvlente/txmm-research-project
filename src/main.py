from data import TranslationMatrixCorpus
from evaluate import evaluate
import features


corpus = TranslationMatrixCorpus('../data/corpus1', ['aeneis', 'oresteia'], ['gk', 'mdhs'])

experiments = [
    [features.SemanticBaseline()],
    [features.SentenceCount()],
    [features.AlphabetFrequency()],
    [features.FrequentWords(100)],
    [features.FrequentWords(100, exclude_stopwords=True)],
    [features.CharacterNGrams(2, 100)],
    [features.CharacterNGrams(3, 100)],
    [features.POS()],
]

if __name__ == '__main__':
    for fts in experiments:
        print('Evaluating feature set [', ', '.join(str(f) for f in fts), ']')
        results = evaluate(corpus, feature_extractors=fts, report=True)
        print('Results:')
        print('author    ', results['author'])
        print('translator', results['translator'])
        print('')
