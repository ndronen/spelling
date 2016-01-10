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
    def __init__(self, words=None, distances=[1]):
        self.__dict__.update(locals())
        del self.self

    def run(self):
        corpus = []
        pbar = build_progressbar(self.words)
        for i,word in enumerate(self.words):
            pbar.update(i+1)
            for d in self.distances:
                # Make this a set, because typo_generator doesn't
                # guarantee uniqueness.
                typos = set([t for t in typo_generator(word, d)])
                for typo in typos:
                    if typo == word:
                        continue
                    corpus.append((word,typo,d))
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
