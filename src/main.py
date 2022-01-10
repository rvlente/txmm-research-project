from data import TranslationMatrixCorpus
from evaluate import evaluate
import features


corpus1 = TranslationMatrixCorpus('../data/corpus1', ['aeneis', 'oresteia'], ['gk', 'mdhs'], 500)
corpus2 = TranslationMatrixCorpus('../data/corpus2', ['mjortvyje-dushi', 'otcy-i-deti'], ['cg', 'cjh'], 4000)

experiments1 = [
    [features.SemanticBaseline('I')],
    [features.SentenceCount()],
    [features.AlphabetFrequency()],
    [features.FrequentWords(100)],
    [features.FrequentWords(100,  exclude_stopwords='nl')],
    [features.CharacterNGrams(2, 100)],
    [features.CharacterNGrams(3, 100)],
    [features.POS(lang='nl')],
]

experiments2 = [
    [features.SemanticBaseline('II')],
    [features.SentenceCount()],
    [features.AlphabetFrequency()],
    [features.FrequentWords(100)],
    [features.FrequentWords(100, exclude_stopwords='en')],
    [features.CharacterNGrams(2, 100)],
    [features.CharacterNGrams(3, 100)],
    [features.POS(lang='en')],
]

if __name__ == '__main__':
    print('Corpus I')
    for fts in experiments1:
        print('Evaluating feature set [', ', '.join(str(f) for f in fts), ']')
        results = evaluate(corpus1, feature_extractors=fts, report=True)
        print('Results:')
        print('author    ', results['author'])
        print('translator', results['translator'])
        print('')

    print('Corpus II')
    for fts in experiments2:
        print('Evaluating feature set [', ', '.join(str(f) for f in fts), ']')
        results = evaluate(corpus2, feature_extractors=fts, report=True)
        print('Results:')
        print('author    ', results['author'])
        print('translator', results['translator'])
        print('')
