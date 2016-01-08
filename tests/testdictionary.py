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
        self.assertTrue(d.check("quick"))
        suggestions = d.suggest("quickq")
        self.assertTrue("quick" in suggestions)

    def test_norvig_without_norvig_language_model(self):
        d = spelling.dictionary.NorvigWithoutNorvigLanguageModel()
        self.assertTrue(d.check("quick"))
        suggestions = d.suggest("quickq")
        self.assertTrue("quick" in suggestions)

    def test_aspell_with_norvig_language_model(self):
        d = spelling.dictionary.AspellWithNorvigLanguageModel()
        self.assertTrue(d.check("quick"))
        suggestions = d.suggest("quickq")
        self.assertTrue("quick" in suggestions)

    def test_norvig_with_aspell_dict_google_language_model(self):
        d = spelling.dictionary.NorvigWithAspellDictGoogleLanguageModel()
        self.assertTrue(d.check("quick"))
        suggestions = d.suggest("quickq")
        self.assertTrue("quick" in suggestions)

if __name__ == '__main__':
    unittest.main()
