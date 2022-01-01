from data import TextTranslationMatrix
from features import alphabet_frequency, pos_frequency


t = TextTranslationMatrix('../data', ['aeneis', 'oresteia'], ['gk', 'mdhs'])

evaluation1 = t.cross_validate(label_on='text', feature_extractors=[
    alphabet_frequency,
    pos_frequency
])

evaluation2 = t.cross_validate(label_on='translator', feature_extractors=[
    alphabet_frequency,
    pos_frequency
])
