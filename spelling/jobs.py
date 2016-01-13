from __future__ import print_function

import sys
import operator
import progressbar
import re
import numpy as np
import pandas as pd
from nltk.stem import SnowballStemmer

from spelling.mitton import build_dataset
from spelling.dictionary import Aspell
from spelling.utils import build_progressbar
from spelling.typodistance import typo_generator
from spelling import errors

from tqdm import tqdm
from spelling.utils import build_progressbar

class Job(object):
    def __init__(self, **kwargs):
        pass

    def run(self):
        raise NotImplementedError()

class KeyboardDistanceCorpus(Job):
    """
    >>> import codecs
    >>> import pandas as pd
    >>> import spelling.mitton
    >>> from spelling.jobs import KeyboardDistanceCorpus
    >>> 
    >>> corpora = [
    >>>         'data/aspell.dat', 'data/birbeck.dat',
    >>>         'data/holbrook-missp.dat', 'data/norvig.dat',
    >>>         'data/wikipedia.dat'
    >>>     ]
    >>> vocabulary = []
    >>> for corpus in corpora:
    >>>     words = spelling.mitton.load_mitton_words(corpus)
    >>>     words = [w[1:] for w in words if w.startswith('$')]
    >>>     vocabulary.extend(words)
    >>> job = KeyboardDistanceCorpus(words=vocabulary, distances=[1, 2])
    >>> corpus_df = job.run()
    >>> corpus_df.to_csv('/tmp/aspell-dict-distances.csv',
    >>>     index=False, sep='\t', encoding='utf8')
    """
    def __init__(self, words=None, distances=[1], sample='all', seed=17):
        self.__dict__.update(locals())
        sample = sample.replace('-', '_')
        self.sample = getattr(self, 'sample_' + sample)
        self.rng = np.random.RandomState(seed)
        del self.self

    def sample_all(self, word, typo, distance):
        return True

    def sample_inverse(self, word, typo, distance):
        return self.rng.uniform() < 1/float(distance)

    def sample_inverse_square(self, word, typo, distance):
        return self.rng.uniform() < 1/float(distance**2)

    def run(self):
        corpus = []
        pbar = build_progressbar(self.words)
        for i,word in enumerate(self.words):
            pbar.update(i+1)
            for d in self.distances:
                # Make this a set, because typo_generator doesn't
                # guarantee uniqueness.
                typos = set()
                for t in typo_generator(word, d):
                    if t != word:
                        typos.add(t)
                for t in typos:
                    if self.sample(word, t, d):
                        corpus.append((word,t,d))
        pbar.finish()
        print("generated %d errors for %d words" %
                (len(corpus), len(self.words)))
        return pd.DataFrame(data=corpus, columns=['word', 'typo', 'distance'])

class DistanceToNearestStem(Job):
    """
    TODO: add post-processing step to compute edit distance to nearest
    word in dictionary by brute force for those words that get a default
    edit distance of 100.

    >>> import codecs
    >>> from spelling.features import levenshtein_distance as dist
    >>> from spelling.jobs import DistanceToNearestStem
    >>> df = pd.read_csv('data/aspell-dict.csv.gz', sep='\t', encoding='utf8')
    >>> job = DistanceToNearestStem(df.word, dist)
    >>> df = job.run()
    >>> df.to_csv('data/aspell-dict-distances.csv.gz',
    >>>     index=False, sep='\t', encoding='utf8')
    """
    def __init__(self, words=None, distance=None, dictionary=Aspell(),
            stemmer=SnowballStemmer("english")):
        self.__dict__.update(locals())
        del self.self

    def run(self):
        """
        words : list
            The words for which to find 
        distance : callable
            Function taking two words and returning a distance.
        dictionary : spelling.dictionary 
            Instance of a class in spelling.dictionary.
        """
        nearest = []
        pbar = build_progressbar(self.words)
        for i,word in enumerate(self.words):
            pbar.update(i+1)
            suggestions = self.suggest(word)
            if len(suggestions) == 0:
                nearest.append((word, "", 100))
            else:
                distances = [(word, s, self.distance(word, s))
                    for s in suggestions]
                sorted_distances = sorted(distances,
                    key=operator.itemgetter(2))
                nearest.append(sorted_distances[0])
        pbar.finish()
        return pd.DataFrame(data=nearest,
            columns=['word', 'suggestion', 'distance'])

    def suggest(self, word):
        """
        Get suggested replacements for a given word from a dictionary.
        Remove words that have the same stem or are otherwise too similar.
        """
        stemmed_word = self.stemmer.stem(word)
        suggestions = []
        for suggestion in self.dictionary.suggest(word):
            if suggestion == word:
                continue

            if self.stemmer.stem(suggestion) == stemmed_word:
                continue

            if ' ' in suggestion or '-' in suggestion:
                # Ignore two-word and hyphenated suggestions.
                continue

            suggestions.append(suggestion)
        return suggestions

class ErrorExtractionJob(Job):

    def __init__(self, words_to_mutate, dictionary, corpus_fn, whitelist_fns=None):
        self.__dict__.update(locals())
        del self.self

    def run(self):
        whitelist = set()
        if self.whitelist_fns is not None:
            for fn in tqdm(self.whitelist_fns):
                with open(fn,"r") as f:
                    for line in f:
                        whitelist.add(line.lower()[:-1])

        injector = errors.ErrorInjector(self.corpus_fn, self.dictionary, whitelist=whitelist)
        result = []
        for word in self.words_to_mutate:
            result.extend(injector.inject_errors(word))
        return result, injector.get_trigrams()

class SplitCSVDataset(Job):
    def __init__(self, input_csv, output_csv):
        self.__dict__.update(locals())

    def run(self):
        df = pd.read_csv(self.input_csv, sep='\t', encoding='utf8')
        unique_words = df.word.unique()
        pbar = build_progressbar(unique_words)
        for i, word in enumerate(unique_words):
            pbar.update(i+1)
            df_tmp = df[df.word == word]
            df_tmp.to_csv(self.output_csv % i,
                    sep='\t', index=False, encoding='utf8')
        pbar.finish()

class BuildDatasetFromCSV(Job):
    def __init__(self, input_csv, output_csv):
        self.__dict__.update(locals())

    def run(self):
        df = pd.read_csv(self.input_csv, sep='\t', encoding='utf8')
        pairs = zip(df.error, df.word)
        dataset = build_dataset(pairs, Aspell())
        dataset.to_csv(self.output_csv, sep='\t', index=False,
                encoding='utf8')
