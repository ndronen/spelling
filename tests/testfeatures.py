# -*- coding: utf-8 -*-
import unittest
from spelling.features import *

class TestTypoDistance(unittest.TestCase):
    def test_soundex(self):
        self.assertEqual("F300", soundex("food"))
        self.assertEqual("0000", soundex(""))

    def test_soundex_levenshtein_distance(self):
        self.assertEqual(0,
                soundex_levenshtein_distance("food", "food"))
        self.assertTrue(
                soundex_levenshtein_distance("food", "axis") > 1)

    def test_metaphone(self):
        self.assertEqual("FT", metaphone("food"))
        self.assertEqual("", metaphone(""))

    def test_metaphone_levenshtein_distance(self):
        self.assertEqual(0,
                metaphone_levenshtein_distance("food", "food"))
        self.assertTrue(
                metaphone_levenshtein_distance("food", "axis") > 1)

    def test_soundex_set_distance(self):
        self.assertEqual(0., soundex_set_distance("food", "food"))
        self.assertEqual(0., soundex_set_distance("tradegy", "tragedy"))
        self.assertEqual(.25, soundex_set_distance("tradegy", "trade"))
        self.assertEqual(1., soundex_set_distance("", "tragedy"))
        self.assertEqual(0., soundex_set_distance("", ""))

    def test_metaphone_set_distance(self):
        self.assertEqual(0., metaphone_set_distance("food", "food"))
        self.assertEqual(0., metaphone_set_distance("tradegy", "tragedy"))
        self.assertTrue(np.allclose(1 - 2/3., metaphone_set_distance("tradegy", "trade")))
        self.assertEqual(3., metaphone_set_distance("", "tragedy"))
        self.assertEqual(0., metaphone_set_distance("", ""))

if __name__ == '__main__':
    unittest.main()
