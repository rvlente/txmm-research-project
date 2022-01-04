from data import TranslationMatrixCorpus
from evaluate import evaluate
import features

corpus = TranslationMatrixCorpus('../data', ['aeneis', 'oresteia'], ['gk', 'mdhs'])


print(evaluate(corpus, label_on='text', feature_extractors=[features.AlphabetCount()]))
print(evaluate(corpus, label_on='translator', feature_extractors=[features.AlphabetCount()]))
