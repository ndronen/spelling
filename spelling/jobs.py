import operator
import progressbar
import re
import pandas as pd
from nltk.stem import SnowballStemmer

from spelling.dictionary import Aspell
from spelling.utils import build_progressbar
from spelling.typodistance import typo_generator

class Job(object):
    def __init__(self, **kwargs):
        pass

    def run(self):
        raise NotImplementedError()

class KeyboardDistanceCorpus(Job):
    """
    >> import codecs
    >> from spelling.jobs import KeyboardDistanceCorpus
    >> df = pd.read_csv('data/aspell-dict.csv.gz', sep='\t', encoding='utf8')
    >> job = KeyboardDistanceCorpus(words=df.word)
    >> corpus_df = job.run()
    """
    def __init__(self, words=None, distances=[1]):
        self.__dict__.update(locals())
        del self.self

    def run(self):
        corpus = []
        pbar = build_progressbar(self.words)
        n = 0
        for i,word in enumerate(self.words):
            pbar.update(i+1)
            for d in self.distances:
                typos = [t for t in typo_generator(word, d)]
                n += len(set(typos))
                for typo in typos:
                    corpus.append((word,typo))
        pbar.finish()
        print("generated %d errors for %d words" %
                (len(corpus), len(self.words)))
        return pd.DataFrame(data=corpus, columns=['word', 'typo'])

class DistanceToNearestStem(Job):
    """
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
