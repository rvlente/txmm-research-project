from data import TextTranslationMatrix
import features

t = TextTranslationMatrix('../data', ['aeneis', 'oresteia'], ['gk', 'mdhs'])

print(t.cross_validate(label_on='text', feature_extractors=[features.AlphabetCount()]))
print(t.cross_validate(label_on='translator', feature_extractors=[features.AlphabetCount()]))
