import sys
if sys.version_info.major == 3:
    unicode = str
import os
import string
import re
import codecs
import collections
import operator
import enchant
import gzip
import numpy as np
import pandas as pd
import threading
import pickle

import jellyfish
import spelling.preprocess

from sklearn.neighbors import NearestNeighbors, LSHForest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import check_random_state

from .utils import build_progressbar as build_pbar

NORVIG_DATA_PATH='data/big.txt.gz'
ASPELL_DATA_PATH='data/aspell-dict.csv.gz'

class Word(object):
    def __init__(self, token, indexer=lambda word: word):
        assert isinstance(token, unicode)
        self.__dict__.update(locals())
        del self.self

    @property
    def key(self):
        return self.indexer(self.token)

    @staticmethod
    def build(x):
        if isinstance(x, Word):
            return x
        else:
            return Word(x)


###########################################################################
# Classes for retrieving candidates.
###########################################################################


class HashBucketRetriever(dict):
    def __init__(self, vocabulary, hasher):
        self.hasher = hasher
        self.phone_to_word = collections.defaultdict(list)
        for word in vocabulary:
            word = unicode(word)
            self.phone_to_word[self.hasher(word)].append(word)

    def __getitem__(self, word):
        assert isinstance(word, unicode)
        return self.phone_to_word[self.hasher(word)]


class RandomRetriever(dict):
    def __init__(self, vocabulary, n_candidates, random_state=17):
        self.vocabulary = list(vocabulary)
        self.n_candidates = n_candidates
        self.random_state = check_random_state(random_state)

    def __getitem__(self, word):
        return self.random_state.choice(self.vocabulary,
                size=self.n_candidates, replace=False)

class EditDistanceRetriever(dict):
    def __init__(self, vocabulary, alphabet=string.ascii_lowercase, stop_retrieving_when_found=True):
        self.vocabulary = set(vocabulary)
        self.alphabet = alphabet
        self.stop_retrieving_when_found = stop_retrieving_when_found

    def edits1(self, word):
        splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes    = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
        replaces   = [a + c + b[1:] for a, b in splits for c in self.alphabet if b]
        inserts    = [a + c + b     for a, b in splits for c in self.alphabet]
        return set(deletes + transposes + replaces + inserts)

    def known_edits2(self, word):
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1) if e2 in self.vocabulary)

    def known(self, words):
        return set(w for w in words if w in self.vocabulary)

    def __getitem__(self, word):
        if self.stop_retrieving_when_found:
            candidates = self.known([word]) or \
                    self.known(self.edits1(word)) or \
                    self.known_edits2(word) or \
                    [word]
        else:
            candidates = []
            candidates.extend(self.known([word]))
            candidates.extend(self.known(self.edits1(word)))
            candidates.extend(self.known_edits2(word))
        return list(candidates)

class NearestNeighborsRetriever(dict):
    def __init__(self, vocabulary, estimator, ngram_range=(1,1)):
        self.__dict__.update(locals())
        del self.self
        assert isinstance(vocabulary, (list, tuple))
        self.vocabulary = np.array(vocabulary)

        if ngram_range[0] == 1 and ngram_range[1] == 1:
            x, _ = spelling.preprocess.build_char_matrix(vocabulary)
        else:
            self.cv = CountVectorizer(ngram_range=ngram_range, analyzer='char_wb')
            x = self.cv.fit_transform(vocabulary).todense()

        self.estimator.fit(x)

    def retrieve(self, word, n_neighbors=None):
        assert isinstance(word, unicode)

        if n_neighbors is None:
            n_candidates = self.estimator.n_neighbors

        if hasattr(self, 'cv'):
            wordx = self.cv.transform([word]).todense()
        else:
            wordx, _ = spelling.preprocess.build_char_matrix([word])

        idx = self.estimator.kneighbors(wordx, n_neighbors=n_candidates, return_distance=False)
        try:
            return self.vocabulary[idx[0]]
        except IndexError as e:
            print(e)
            print(self.vocabulary.shape)
            print(type(idx))
            print(idx.shape)
            print(idx[0])
            print(idx[0].shape)
            raise e

        return candidates

    def __getitem__(self, word):
        return self.retrieve(word)


class AspellRetriever(dict):
    def __init__(self, lang='en_US', reentrant=False):
        self.dictionary = enchant.Dict(lang)
        self.reentrant = reentrant
        if reentrant:
            self.lock = threading.Lock()

    def __getitem__(self, word):
        if self.reentrant:
            with self.lock:
                return self.dictionary.suggest(word)
        else:
            return self.dictionary.suggest(word)


class RetrieverCollection(dict):
    def __init__(self, retrievers):
        self.retrievers = retrievers

    def __getitem__(self, word):
        words = set()
        for r in self.retrievers:
            words.update(r[word])
        return list(words)

class CachingRetriever(dict):
    def __init__(self, retriever, cache_dir='/tmp/dictcache/'):
        self.retriever = retriever
        self.cache_dir = cache_dir
        self.cache = {}

        try:
            os.mkdir(cache_dir)
        except FileExistsError:
            pass

    def load_cache(self, verbose=False):
        cache_files = os.listdir(self.cache_dir)
        pbar = None
        if verbose:
            pbar = build_pbar(cache_files)
        for i,cache_file in enumerate(cache_files):
            if verbose:
                pbar.update(i+1)
            word = os.path.basename(cache_file)
            self[word]
        if verbose:
            pbar.finish()

    def __getitem__(self, word):
        if word in self.cache:
            return self.cache[word]

        cache_file = self.cache_dir + '/' + word
        if os.path.exists(cache_file):
            try:
                words = pickle.load(open(cache_file, 'rb'))
            except (EOFError,IOError) as load_exception:
                words = self.retriever[word]

                print('Caught an opening cache file %s.  Will remove %s' % (
                    cache_file, cache_file))
                try:
                   os.remove(cache_file) 
                except Exception as remove_exception:
                    pass
            self.cache[word] = words
        else:
            words = self.retriever[word]
            pickle.dump(words, open(cache_file, 'wb'))
            self.cache[word] = words

        return self.cache[word]

class FilteringRetriever(dict):
    def __init__(self, retriever, fltr):
        self.__dict__.update(locals())
        del self.self

    def __getitem__(self, word):
        words = self.retriever[word]
        return filter(self.fltr, words)

###########################################################################
# Classes for sorting candidates.
###########################################################################

class Sorter(object):
    def sort(self, word, candidates):
        raise NotImplementedError()

class DistanceSorter(Sorter):
    def __init__(self, distance, reverse=None):
        if callable(distance):
            self.distance = distance
        else:
            self.distance = {
                    'damerau_levenshtein_distance': jellyfish.damerau_levenshtein_distance,
                    'levenshtein_distance': jellyfish.levenshtein_distance,
                    'hamming_distance': jellyfish.hamming_distance,
                    'jaro_winkler': jellyfish.jaro_winkler,
                    'jaro_distance': jellyfish.jaro_distance
                    }[distance]

        if reverse is not None:
            assert isinstance(reverse, bool)
            self.reverse = reverse
        else:
            try:
                self.reverse = {
                        'damerau_levenshtein_distance': False,
                        'levenshtein_distance': False,
                        'hamming_distance': False,
                        'jaro_winkler': True,
                        'jaro_distance': True
                        }[self.distance.__name__]
            except KeyError:
                raise ValueError("'reverse' parameter is required when passing your own distance function")
        
    def sort(self, word, candidates):
        return sorted(set(candidates),
            key=lambda c: self.distance(c, word),
            reverse=self.reverse)


class LanguageModelSorter(Sorter):
    def __init__(self, words, probs):
        self.model = collections.defaultdict(int)
        for i, word in enumerate(words):
            self.model[word] = probs[i]

    def sort(self, word, candidates):
    	return sorted(candidates, key=self.model.get, reverse=True)

class NonSortingSorter(Sorter):
    def sort(self, word, candidates):
        return list(candidates)

class SortingRetriever(dict):
    def __init__(self, retriever, sorter):
        self.__dict__.update(locals())
        del self.self

    def __getitem__(self, word):
        candidates = self.retriever[word]
        return self.sorter.sort(word, candidates)

class TopKRetriever(dict):
    def __init__(self, retriever, k):
        self.retriever = retriever
        self.k = k

    def __getitem__(self, word):
        return self.retriever[word][:self.k]

class BottomKRetriever(dict):
    def __init__(self, retriever, k):
        self.retriever = retriever
        self.k = k

    def __getitem__(self, word):
        return self.retriever[word][-self.k:]

###########################################################################
# Dictionary implementations.
###########################################################################


class Dictionary(object):
    def check(self, word):
        raise NotImplementedError()

    def suggest(self, word):
        raise NotImplementedError()

    def correct(self, word):
        raise NotImplementedError()


class Aspell(Dictionary):
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
        suggestions = []
        for s in self.dictionary.suggest(word):
            if " " in s or "-" in s:
                continue
            suggestions.append(s)
        return suggestions


class ModularDictionary(Dictionary):
    def __init__(self, vocabulary, retrievers, sorter, filters=[]):
        self.__dict__.update(locals())
        del self.self

    def check(self, word):
        return word in self.vocabulary

    def suggest(self, word):
        candidates = list()
        seen = set()
        for r in self.retrievers:
            for candidate in r[word]:
                if all([f(candidate) for f in self.filters]):
                    if candidate not in seen:
                        candidates.append(candidate)
                        seen.add(candidate)
        return self.sorter.sort(word, candidates)

    def correct(self, word):
        return self.suggest(word)[0]


###########################################################################
# Dictionary builders
###########################################################################


class DictionaryBuilder(object):
    def build(**kwargs):
        raise NotImplementedError()


class ModularDictionaryBuilder(DictionaryBuilder):
    def __init__(self):
        self.vocabulary = []
        self.retrievers = []
        self.sorter = NonSortingSorter()
        self.filters = []
        self.probs = []

    def with_vocabulary(self, vocab_type, data_path):
        self.vocabulary = []
        self.probs = []

        if vocab_type == 'norvig':
            # Load norvig vocabulary and frequencies.
            train_file = gzip.open(data_path)
            train_file = codecs.EncodedFile(train_file, 'utf8')

            text = str(train_file.read())
            words = re.findall('[a-z]+', text.lower())

            model = collections.defaultdict(lambda: 1)
            for w in words:
                model[w] += 1

            for k,v in model.items():
                self.vocabulary.append(k)
                self.probs.append(v)
        elif vocab_type == 'aspell':
            # Load aspell vocabulary and frequencies.
            df = pd.read_csv(data_path, sep='\t', encoding='utf8')
            self.vocabulary = df.word.tolist()
            self.probs = df.google_ngram_prob
        else:
            raise ValueError('unknown vocabulary type %s; use "norvig" or "aspell"')

    def with_sorter(self, sorter_type, **kwargs):
        if sorter_type == 'lm':
            try:
                self.sorter = LanguageModelSorter(self.vocabulary, self.probs)
            except AttributeError:
                raise RuntimeError("call with_language_model(type) before calling with_language_model_sorter")
        elif sorter_type == 'distance':
            self.sorter = DistanceSorter(kwargs['distance'])
        else:
            raise ValueError('unknown sorter type %s; use "lm" or "distance"')

    def add_retriever(self, retriever_type, **kwargs):
        if retriever_type == 'editdistance':
            self.retrievers.append(EditDistanceRetriever(self.vocabulary))
        elif retriever_type == 'hashbucket':
            self.retrievers.append(HashBucketRetriever(**kwargs))
        elif retriever_type == 'aspell':
            self.retrievers.append(AspellRetriever())
        elif retriever_type == 'neighbor':
            self.retrievers.append(NearestNeighborsRetriever(**kwargs))
        else:
            raise ValueError('unknown retriever type "%s"; use "editdistance", "hashbucket", "aspell", or "neighbor"' % retriever_type)

    def add_filter(self, f):
        self.filters.append(f)

    def build(self):
    	return ModularDictionary(self.vocabulary, self.retrievers,
            self.sorter, filters=self.filters)

def build_norvig():
    builder = ModularDictionaryBuilder()
    builder.with_vocabulary('norvig', data_path=NORVIG_DATA_PATH)
    builder.with_sorter('lm')
    builder.add_retriever('editdistance')
    return builder.build()

def build_norvig_without_language_model():
    builder = ModularDictionaryBuilder()
    builder.with_vocabulary('norvig', data_path=NORVIG_DATA_PATH)
    builder.add_retriever('editdistance')
    return builder.build()

def build_norvig_with_aspell_vocab_without_language_model():
    builder = ModularDictionaryBuilder()
    builder.with_vocabulary('aspell', data_path=ASPELL_DATA_PATH)
    builder.add_retriever('editdistance')
    return builder.build()

def build_norvig_with_aspell_vocab_with_language_model_sorter():
    builder = ModularDictionaryBuilder()
    builder.with_vocabulary('aspell', data_path=ASPELL_DATA_PATH)
    builder.add_retriever('editdistance')
    builder.with_sorter('lm')
    return builder.build()

def build_aspell():
    builder = ModularDictionaryBuilder()
    builder.with_vocabulary('aspell', data_path=ASPELL_DATA_PATH)
    builder.add_retriever('aspell')
    return builder.build()

def build_aspell_with_language_model_sorter():
    builder = ModularDictionaryBuilder()
    builder.with_vocabulary('aspell', data_path=ASPELL_DATA_PATH)
    builder.add_retriever('aspell')
    builder.with_sorter('lm')
    return builder.build()

def build_aspell_with_jaro_winkler_sorter():
    builder = ModularDictionaryBuilder()
    builder.with_vocabulary('aspell', data_path=ASPELL_DATA_PATH)
    builder.add_retriever('aspell')
    builder.with_sorter('distance', distance='jaro_winkler')
    return builder.build()

def build_aspell_and_edit_distance_retriever_with_jaro_winkler_sorter():
    builder = ModularDictionaryBuilder()
    builder.with_vocabulary('aspell', data_path=ASPELL_DATA_PATH)
    builder.add_retriever('aspell')
    builder.add_retriever('editdistance')
    builder.with_sorter('distance', distance='jaro_winkler')
    return builder.build()

def build_aspell_vocab_with_metaphone_retriever_and_language_model_sorter():
    builder = ModularDictionaryBuilder()
    builder.with_vocabulary('aspell', data_path=ASPELL_DATA_PATH)
    builder.add_retriever('hashbucket',
            vocabulary=builder.vocabulary, hasher=jellyfish.metaphone)
    builder.with_sorter('lm')
    return builder.build()

def build_aspell_vocab_with_nn_retriever_and_language_model_sorter():
    estimator = NearestNeighbors(n_neighbors=20, metric='hamming', algorithm='auto')

    builder = ModularDictionaryBuilder()
    builder.with_vocabulary('aspell', data_path=ASPELL_DATA_PATH)
    builder.add_retriever('neighbor', 
            vocabulary=builder.vocabulary, estimator=estimator)
    builder.with_sorter('lm')
    return builder.build()
