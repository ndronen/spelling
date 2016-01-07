# -*- coding: utf-8 -*-
from __future__ import absolute_import

import sys
import os
import numpy as np

import unittest
import spelling.dictionary
from spelling.dictionary import NORVIG_DATA_PATH as train_path

class TestDictionary(unittest.TestCase):
    def test_enchant(self):
        d = spelling.dictionary.Enchant()
        self.assertTrue(d.check("quick"))
        suggestions = d.suggest("quickq")
        self.assertTrue("quick" in suggestions)

    def test_norvig(self):
        d = spelling.dictionary.Norvig(train_path=train_path)
        self.assertTrue(d.check("quick"))
        suggestions = d.suggest("quickq")
        self.assertTrue("quick" in suggestions)

    def test_norvig_without_language_model(self):
        d = spelling.dictionary.NorvigWithoutLanguageModel(train_path=train_path)
        self.assertTrue(d.check("quick"))
        suggestions = d.suggest("quickq")
        self.assertTrue("quick" in suggestions)

    def test_enchant_with_language_model(self):
        d = spelling.dictionary.EnchantWithLanguageModel(train_path=train_path)
        self.assertTrue(d.check("quick"))
        suggestions = d.suggest("quickq")
        self.assertTrue("quick" in suggestions)

if __name__ == '__main__':
    unittest.main()
