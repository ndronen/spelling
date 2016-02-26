# -*- coding: utf-8 -*-
from __future__ import absolute_import

import string
import itertools
import sys
import os
import numpy as np
import jellyfish

import unittest
import spelling.dictionary
from spelling.dictionary import NORVIG_DATA_PATH as train_path
from spelling.features import metaphone

import sklearn.neighbors

class TestDictionaryWord(unittest.TestCase):
    def test_dictionary_word(self):
        token = u"quick"

        word = spelling.dictionary.Word(token)
        self.assertEqual(token, word.token)
        self.assertEqual(token, word.key)

        word = spelling.dictionary.Word(token, indexer=metaphone)
        self.assertEqual(token, word.token)
        self.assertEqual(metaphone(token), word.key)

class TestDistanceSorter(unittest.TestCase):
    def test_candidate_sorter(self):
        words = [u"ax", u"bxc", u"ay", u"byc", u"abracadabra"]
        word = u"axial"
        for distance in ['levenshtein_distance', 'jaro_distance', 'jaro_winkler', 'damerau_levenshtein_distance']:
            sorter = spelling.dictionary.DistanceSorter(distance)
            self.assertEqual(len(words), len(sorter.sort(word, words)))
            self.assertEqual(words[0], sorter.sort(word, words)[0])

        for distance in [jellyfish.levenshtein_distance, jellyfish.jaro_distance, jellyfish.jaro_winkler, jellyfish.damerau_levenshtein_distance]:
            sorter = spelling.dictionary.DistanceSorter(distance)
            self.assertEqual(len(words), len(sorter.sort(word, words)))
            self.assertEqual(words[0], sorter.sort(word, words)[0])

class TestRetrievers(unittest.TestCase):
    def test_nearest_neighbors_retriever_nn(self):
        words = [u"elf", u"self", u"elves", u"a", u"b", u"c", u"d"]
        estimator = sklearn.neighbors.NearestNeighbors(metric='hamming', algorithm='auto',
                n_neighbors=int(len(words)/2.))
        r = spelling.dictionary.NearestNeighborsRetriever(words, estimator)
        candidates = r[u"xelf"]
        self.assertEqual(estimator.n_neighbors, len(candidates))

    def test_nearest_neighbors_retriever_lsh(self):
        words = [u"elf", u"self", u"elves", u"a", u"b", u"c", u"d"]
        words.extend([unicode(s) for s in itertools.combinations(string.ascii_letters, 4)])
        estimator = sklearn.neighbors.LSHForest(n_neighbors=int(len(words)/2.))
        r = spelling.dictionary.NearestNeighborsRetriever(words, estimator, ngram_range=(2,2))
        candidates = r[u"xelf"]
        self.assertEqual(estimator.n_neighbors, len(candidates))

    def test_edit_distance_retriever(self):
        words = [u"elf", u"self", u"elves", u"a", u"b", u"c", u"d"]
        r = spelling.dictionary.EditDistanceRetriever(words)
        candidates = r[u"xelf"]
        self.assertEqual(2, len(candidates))
        self.assertTrue(all([w in candidates for w in words[0:1]]))


class TestDictionary(unittest.TestCase):
    def test_aspell(self):
        d = spelling.dictionary.build_aspell()
        self.assertTrue(d.check(u"quick"))
        suggestions = d.suggest(u"quickq")
        self.assertTrue(u"quick" in suggestions)

    def test_norvig(self):
        d = spelling.dictionary.build_norvig()
        self.assertTrue(d.check(u"american"))
        suggestions = d.suggest(u"amaricae")
        known_suggestions = [u'american', u'america', u'avarice']
        self.assertEqual(len(known_suggestions), len(suggestions))
        for s in known_suggestions:
            self.assertTrue(s in suggestions)

    def test_norviglanguage_model(self):
        d = spelling.dictionary.build_norvig_without_language_model()
        self.assertTrue(d.check(u"quick"))
        suggestions = d.suggest(u"quickq")
        self.assertTrue(u"quick" in suggestions)

    def test_norvig_with_aspell_vocab_with_language_model_sorter(self):
        d = spelling.dictionary.build_norvig_with_aspell_vocab_with_language_model_sorter()
        self.assertTrue(d.check(u"quick"))
        suggestions = d.suggest(u"quickq")
        self.assertTrue(u"quick" in suggestions)

    def test_norvig_with_aspell_vocab_without_language_model(self):
        d = spelling.dictionary.build_norvig_with_aspell_vocab_without_language_model()
        self.assertTrue(d.check(u"quick"))
        suggestions = d.suggest(u"quickq")
        self.assertTrue(u"quick" in suggestions)

    def test_aspell_vocab_with_metaphone_retriever_and_language_model_sorter(self):
        d = spelling.dictionary.build_aspell_vocab_with_metaphone_retriever_and_language_model_sorter()
        self.assertTrue(d.check(u"quick"))
        suggestions = d.suggest(u"farz")
        self.assertTrue(u"various" in suggestions)

        word = u"Amarkia" # "America"
        self.assertTrue(u"America" in d.suggest(word))

    def test_aspell_vocab_with_nn_retriever_and_language_model_sorter(self):
        d = spelling.dictionary.build_aspell_vocab_with_nn_retriever_and_language_model_sorter()
        self.assertTrue(d.check(u"quick"))
        suggestions = d.suggest(u"quickq")
        self.assertTrue(u"quick" in suggestions)

if __name__ == '__main__':
    unittest.main()
