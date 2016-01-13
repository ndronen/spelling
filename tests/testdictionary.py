# -*- coding: utf-8 -*-
from __future__ import absolute_import

import sys
import os
import numpy as np

import unittest
import spelling.dictionary
from spelling.dictionary import NORVIG_DATA_PATH as train_path

class TestDictionary(unittest.TestCase):
    def test_aspell(self):
        d = spelling.dictionary.Aspell()
        self.assertTrue(d.check("quick"))
        suggestions = d.suggest("quickq")
        self.assertTrue("quick" in suggestions)

    def test_norvig(self):
        d = spelling.dictionary.Norvig()
        self.assertTrue(d.check("american"))
        suggestions = d.suggest("amaricae")
        known_suggestions = ['american', 'america', 'avarice']
        self.assertEqual(len(known_suggestions), len(suggestions))
        for s in known_suggestions:
            self.assertTrue(s in suggestions)

    def test_norvig_without_language_model(self):
        d = spelling.dictionary.NorvigWithoutLanguageModel()
        self.assertTrue(d.check("quick"))
        suggestions = d.suggest("quickq")
        self.assertTrue("quick" in suggestions)

    def test_aspell_with_norvig_language_model(self):
        d = spelling.dictionary.AspellWithNorvigLanguageModel()
        self.assertTrue(d.check("quick"))
        suggestions = d.suggest("quickq")
        self.assertTrue("quick" in suggestions)

    def test_norvig_with_aspell_dict_with_google_language_model(self):
        d = spelling.dictionary.NorvigWithAspellVocabAndGoogleLanguageModel()
        self.assertTrue(d.check("quick"))
        suggestions = d.suggest("quickq")
        self.assertTrue("quick" in suggestions)

    def test_norvig_with_aspell_dict_without_language_model(self):
        d = spelling.dictionary.NorvigWithAspellVocabWithoutLanguageModel()
        self.assertTrue(d.check("quick"))
        suggestions = d.suggest("quickq")
        self.assertTrue("quick" in suggestions)

    def test_aspell_with_google_language_model(self):
        d = spelling.dictionary.AspellWithGoogleLanguageModel()
        self.assertTrue(d.check("quick"))
        suggestions = d.suggest("quickq")
        self.assertTrue("quick" in suggestions)

    def test_norvig_with_aspell_dict_google_language_model_phonetic_candidates(self):
        d = spelling.dictionary.NorvigWithAspellVocabGoogleLanguageModelPhoneticCandidates()
        self.assertTrue(d.check("quick"))
        suggestions = d.suggest("quickq")
        self.assertTrue("quick" in suggestions)

        word = "Amarkia" # "America"
        d_original = spelling.dictionary.Norvig()
        s_original = d_original.suggest(word)
        s_this = d.suggest(word)
        self.assertTrue(len(s_original) < len(s_this))

if __name__ == '__main__':
    unittest.main()
