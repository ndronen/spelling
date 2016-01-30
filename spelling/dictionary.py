import re
import codecs
import collections
import operator
import enchant
import Levenshtein 
import gzip
import pandas as pd

from spelling.features import metaphone, levenshtein_distance

NORVIG_DATA_PATH='data/big.txt.gz'
ASPELL_DATA_PATH='data/aspell-dict.csv.gz'

class Aspell(object):
    def __init__(self, lang='en_US', train_path=None):
        self.dictionary = enchant.Dict(lang)

    def check(self, word):
        return self.dictionary.check(word)

    def suggest(self, word):
        return self.dictionary.suggest(word)

    def correct(self, word):
        return self.suggest(word)[0]

class AspellUniword(Aspell):
    def suggest(self, word):
        return [word for word in self.dictionary.suggest(word) if " " not in word and "-" not in word]

class Norvig(object):
    """
    Adapted from http://norvig.com/spell-correct.html
    """
    def __init__(self, lang=None, train_path=NORVIG_DATA_PATH):
        train_file = gzip.open(train_path)
        train_file = codecs.EncodedFile(train_file, 'utf8')
        #data = gzip.open(train_path).read()
        self.model = self.train(self.words(train_file.read()))
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
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1) if e2 in self.model)

    def known(self, words):
        return set(w for w in words if w in self.model)

    def build_candidates(self, word):
        return self.known([word]) or self.known(self.edits1(word)) or self.known_edits2(word) or [word]

    def check(self, word):
        return word in self.model

    def suggest(self, word):
        #candidates = self.known([word]) or self.known(self.edits1(word)) or self.known_edits2(word) or [word]
        candidates = self.build_candidates(word)
        return sorted(candidates, key=self.model.get, reverse=True)

    def correct(self, word):
        #candidates = self.known([word]) or self.known(self.edits1(word)) or self.known_edits2(word) or [word]
        candidates = self.build_candidates(word)
        return max(candidates, key=self.model.get)

class NorvigWithoutLanguageModel(Norvig):
    def train(self, features):
        """
        Make the frequency of all words the same, so corrections are
        sorted only by edit distance.
        """
        model = collections.defaultdict(lambda: 1)
        for f in features:
            model[f] = 1
        return model

class NorvigWithAspellVocab(Norvig):
    def __init__(self, lang=None, train_path=ASPELL_DATA_PATH):
        df = pd.read_csv(train_path, sep='\t', encoding='utf8')
        self.model = self.train(df)
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'

    def train(self, df):
        raise NotImplementedError()

class NorvigWithAspellVocabAndGoogleLanguageModel(NorvigWithAspellVocab):
    def train(self, df):
        d = collections.defaultdict(float)
        d.update(dict(zip(df.word, df.google_ngram_prob)))
        return d

class NorvigWithAspellVocabWithoutLanguageModel(NorvigWithAspellVocab):
    def train(self, df):
        return dict(zip(df.word, [1] * len(df)))

class AspellWithNorvigLanguageModel(Norvig):
    def __init__(self, lang='en_US', train_path=NORVIG_DATA_PATH):
        super(AspellWithNorvigLanguageModel, self).__init__(train_path)
        self.dictionary = enchant.Dict(lang)

    def check(self, word):
        return self.dictionary.check(word)

    def suggest(self, word):
        suggestions = self.dictionary.suggest(word)
        return sorted(suggestions, key=lambda s: -self.model[s])

    def correct(self, word):
        suggestions = self.suggest(word)
        return max(suggestions, key=self.model.get)

class AspellWithGoogleLanguageModel(NorvigWithAspellVocabAndGoogleLanguageModel):
    def __init__(self, lang='en_US', train_path=ASPELL_DATA_PATH):
        super(AspellWithGoogleLanguageModel, self).__init__(train_path)
        self.dictionary = enchant.Dict(lang)

    def check(self, word):
        return self.dictionary.check(word)

    def suggest(self, word):
        suggestions = self.dictionary.suggest(word)
        return sorted(suggestions, key=lambda s: -self.model[s])

    def correct(self, word):
        suggestions = self.suggest(word)
        return max(suggestions, key=self.model.get)

class NorvigWithAspellVocabGoogleLanguageModelPhoneticCandidates(NorvigWithAspellVocabAndGoogleLanguageModel):
    """
    The Norvig implementation only considers candidates that are up to
    two edit operations from the unknown word; the number of strings
    that are three or more edit operations is quite large.  This limits
    the dictionary's ability to offer good suggestions, particularly for
    spelling errors that occur due to lack of knowledge of orthographic
    conventions.

    This class augments the Norvig implementation with candidate
    suggestions that are phonetically similar to a given word and possibly
    more than two edit operations away.  Using phonetic hashes in this
    way yields reasonable candidates without the computational burden
    of the brute force approach in the Norvig class.
    """
    def __init__(self, lang=None, train_path=ASPELL_DATA_PATH, hasher=metaphone, max_distance=3):
        super(NorvigWithAspellVocabGoogleLanguageModelPhoneticCandidates, self).__init__(lang, train_path)
        self.phonedict = PhoneticDictionary(self.model.keys(), hasher)
        self.max_distance = max_distance

    def build_candidates(self, word):
        candidates = self.known([word]) or self.known(self.edits1(word)) or self.known_edits2(word) or [word]
        candidates = set(candidates)
        # Eliminate any phonetically similar words that are greater than
        # three edit operations away.
        phonetic_candidates = self.phonedict[word]
        for pc in phonetic_candidates:
            if levenshtein_distance(word, pc) <= self.max_distance:
                candidates.add(pc)
        return candidates

class PhoneticDictionary(dict):
    def __init__(self, vocab, hasher):
        self.hasher = hasher
        self.phone_to_word = collections.defaultdict(list)
        for word in vocab:
            self.phone_to_word[self.hasher(word)].append(word)

    def __getitem__(self, name):
        return self.phone_to_word[self.hasher(name)]
