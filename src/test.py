from data import TextTranslationMatrix
from features import alphabet_frequency, pos_frequency


t = TextTranslationMatrix('../data', ['aeneis', 'oresteia'], ['gk', 'mdhs'])

model1, evaluation1 = t.create_model(label_on='text', feature_extractors=[
    alphabet_frequency,
    pos_frequency
])

model2, evaluation2 = t.create_model(label_on='translator', feature_extractors=[
    alphabet_frequency,
    pos_frequency
])
