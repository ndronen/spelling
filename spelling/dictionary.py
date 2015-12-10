import re
import collections
import operator
import enchant
import Levenshtein 
import gzip

NORVIG_DATA_PATH='data/big.txt.gz'

class Norvig(object):
    """
    Adapted from http://norvig.com/spell-correct.html
    """
    def __init__(self, lang=None, train_path=NORVIG_DATA_PATH):
        self.nwords = self.train(self.words(gzip.open(train_path).read()))
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'

    def words(self, text): return re.findall('[a-z]+', text.lower()) 

    def train(self, features):
        model = collections.defaultdict(lambda: 1)
        for f in features:
            model[f] += 1
        return model

    def edits1(self, word):
        splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes    = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
        replaces   = [a + c + b[1:] for a, b in splits for c in self.alphabet if b]
        inserts    = [a + c + b     for a, b in splits for c in self.alphabet]
        return set(deletes + transposes + replaces + inserts)

    def known_edits2(self, word):
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1) if e2 in self.nwords)

    def known(self, words):
        return set(w for w in words if w in self.nwords)

    def check(self, word):
        return word in self.nwords

    def suggest(self, word):
        candidates = self.known([word]) or self.known(self.edits1(word)) or self.known_edits2(word) or [word]
        return sorted(candidates, key=self.nwords.get, reverse=True)

    def correct(self, word):
        candidates = self.known([word]) or self.known(self.edits1(word)) or self.known_edits2(word) or [word]
        return max(candidates, key=self.nwords.get)

class Enchant(object):
    def __init__(self, lang='en_US', train_path=None):
        self.dictionary = enchant.Dict(lang)

    def check(self, word):
        return self.dictionary.check(word)

    def suggest(self, word):
        return self.dictionary.suggest(word)

    def correct(self, word):
        return self.suggest(word)[0]

class EnchantWithLanguageModel(Norvig):
    def __init__(self, lang='en_US', train_path=NORVIG_DATA_PATH):
        super(EnchantWithLanguageModel, self).__init__(train_path)
        self.dictionary = enchant.Dict(lang)

    def check(self, word):
        return self.dictionary.check(word)

    def suggest(self, word):
        suggestions = self.dictionary.suggest(word)
        return sorted(suggestions, key=lambda s: (Levenshtein.distance(s, word), -self.nwords[s]))

    def correct(self, word):
        suggestions = self.suggest(word)
        return max(suggestions, key=self.nwords.get)
