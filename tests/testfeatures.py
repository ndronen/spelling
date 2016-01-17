# -*- coding: utf-8 -*-
import unittest
from spelling.features import *

class TestTypoDistance(unittest.TestCase):
    def test_soundex(self):
        self.assertEqual(u"F300", soundex(u"food"))
        self.assertEqual(u"", soundex(u""))

    def test_soundex_levenshtein_distance(self):
        metric = "levenshtein"
        encoding = "soundex"
        self.assertEqual(0,
                distance(u'food', u'food', metric, encoding))
        self.assertTrue(
                distance(u'food', u'axis', metric, encoding) > 1)

    def test_metaphone(self):
        self.assertEqual(u"FT", metaphone(u"food"))
        self.assertEqual(u"", metaphone(u""))

    def test_metaphone_levenshtein_distance(self):
        metric = "levenshtein"
        encoding = "metaphone"
        self.assertEqual(0,
                distance(u"food", u"food", metric, encoding))
        self.assertTrue(
                distance(u"food", u"axis", metric, encoding) > 1)

    def test_soundex_set_distance(self):
        metric = "set"
        encoding = "soundex"
        self.assertEqual(0., distance(u"food", u"food", metric, encoding))
        self.assertEqual(0., distance(u"tradegy", u"tragedy", metric, encoding))
        self.assertEqual(.25, distance(u"tradegy", u"trade", metric, encoding))
        self.assertEqual(1., distance(u"", u"tragedy", metric, encoding))
        self.assertEqual(0., distance(u"", u"", metric, encoding))

    def test_metaphone_set_distance(self):
        metric = "set"
        encoding = "metaphone"
        self.assertEqual(0., distance(u"food", u"food", metric, encoding))
        self.assertEqual(0., distance(u"tradegy", u"tragedy", metric, encoding))
        self.assertTrue(.3333, distance(u"trade", u"tradegy", metric, encoding))
        self.assertEqual(1., distance(u"", u"tragedy", metric, encoding))
        self.assertEqual(0., distance(u"", u"", metric, encoding))

if __name__ == '__main__':
    unittest.main()
