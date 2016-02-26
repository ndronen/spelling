from __future__ import print_function

import sys
import collections
import operator
import progressbar
import re
import numpy as np
import pandas as pd
from nltk.stem import SnowballStemmer

from spelling.mitton import (load_mitton_words,
        build_dataset, build_mitton_pairs)
from spelling.dictionary import Aspell
from spelling.utils import build_progressbar
from spelling.typodistance import typo_generator
from spelling import errors
from spelling.edits import EditFinder, subsequences, EditConstraintError
import spelling.baseline

from tqdm import tqdm
from spelling.utils import build_progressbar

class Job(object):
    def run(self):
        raise NotImplementedError()

class KeyboardDistanceCorpus(Job):
    """
    >>> import codecs
    >>> import pandas as pd
    >>> import spelling.mitton
    >>> from spelling.jobs import KeyboardDistanceCorpus
    >>> 
    >>> corpora = spelling.mitton.CORPORA
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
    def __init__(self, words=None, distances=[1], sample='all', max_examples_per_word=sys.maxsize, seed=17):
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
                    if t == word:
                        continue
                    if self.sample(word, t, d):
                        typos.add(t)
                    if len(typos) == self.max_examples_per_word:
                        break
                for t in typos:
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

class BuildLearnedErrorCorpus(Job):
    def __init__(self, real_words, edit_db, enough_errors_for_word, blacklist=set(), max_edits_per_error=1, constraints=[], random_state=17, verbose=0):
        self.__dict__.update(locals())
        del self.self

        if isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        assert callable(self.enough_errors_for_word)

    def run(self):
        errors = []
        pbar = build_progressbar(self.real_words)

        finder = EditFinder()

        for i,word in enumerate(self.real_words):
            pbar.update(i+1)

            # Find all the edits we can make to this word.
            possible_edits = list()
            probs = list()
            for subseq in subsequences(word):
                # Probably delete this if statement as redundant.
                for e in self.edit_db.edits(subseq):
                    _, error_subseq, count = e
                    possible_edit = (subseq, error_subseq)
                    if count > 0:
                        possible_edits.append(possible_edit)
                        probs.append(count)

            if len(possible_edits) == 0:
                continue

            probs = np.array(probs)
            probs = probs / float(probs.sum())

            seen_edits = set()
            errors_for_word = []
            attempts = 0.

            # Try to generate up to the requested number of errors per word.
            while True:
                try:
                    attempts += 1.

                    if self.enough_errors_for_word(word, errors_for_word):
                        # Generated enough errors for this word.
                        break
                    elif attempts > 10 and len(errors_for_word) / attempts < 0.1:
                        # Not finding many errors to apply.  Break out.
                        break

                    # Sample the number of edits.
                    edit_sizes = np.arange(1, self.max_edits_per_error+1)
                    edit_size_probs = 1. / edit_sizes
                    edit_size_probs /= edit_size_probs.sum()
                    size = self.random_state.choice(edit_sizes, size=1, replace=False,
                            p=edit_size_probs)[0]

                    # Sample edits with probability proportional to the edit's frequency.
                    edit_idx = self.random_state.choice(len(probs), size=size, replace=False, p=probs)

                    edit = []
                    for i in edit_idx:
                        pe = possible_edits[i]
                        if pe in seen_edits:
                            continue
                        seen_edits.add(pe)
                        edit.append(pe)

                    if len(edit) == 0:
                        continue
    
                    # Avoid applying edits that result in unlikely errors.
                    for constraint in self.constraints:
                        for e in edit:
                            if constraint(word, e):
                                raise EditConstraintError("can't apply edit %s=>%s to word '%s'" % \
                                        (e[0], e[1], word))

                    error = finder.apply(word, edit)
                    if error in self.blacklist:
                        # Skip blacklisted words (i.e. non-words in a corpus used to generate the
                        # edit patterns in the edit database).
                        continue

                    errors_for_word.append((word, len(possible_edits), edit, error))

                except EditConstraintError as e:
                    if self.verbose:
                        print(e)

            errors.extend(errors_for_word)

        pbar.finish()
    
        return errors


class LanguageModelBaseline(Job):
    """
    >>> import sklearn.metrics
    >>> import spelling.jobs
    >>> data_dir = "~/proj/modeling/data/spelling/experimental/"
    >>> csv_path = data_dir + "non-word-error-detection-experiment-04-generated-negative-examples.csv"
    >>> job = LanguageModelBaseline(csv_path)
    >>> df = job.run()
    >>> print(confusion_matrix(df.binary_target, df.pred))
    """
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path, sep='\t', encoding='utf8')

    def run(self):
        non_words = self.df[df.binary_target == 0].word.tolist()
        real_words = self.df[df.binary_target == 1].word.tolist()

        non_word_lm = spelling.baseline.CharacterLanguageModel('witten-bell', 3)
        real_word_lm = spelling.baseline.CharacterLanguageModel('witten-bell', 3)
        non_word_lm.fit(non_words)
        real_word_lm.fit(real_words)

        clf = spelling.baseline.LanguageModelClassifier([non_word_lm, real_word_lm])

        words = non_words + real_words
        binary_target = [0] * len(non_words) + [1] * len(real_words)

        pred, prob = clf.predict(words, return_proba=True)

        return pd.DataFrame({
            'word': words, 'binary_target': binary_target,
            'pred': pred, 'prob0': prob[:, 0], 'prob1': prob[:, 1]
            })
