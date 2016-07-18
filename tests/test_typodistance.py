# -*- coding: utf-8 -*-
import unittest
from spelling.typodistance import *

class TestTypoDistance(unittest.TestCase):
    def test_delete(self):
        self.assertEqual(0., typo_distance("two", "two"))
        self.assertEqual(1., typo_distance("two", "to"))
        self.assertEqual(2., typo_distance("two", "t"))

    def test_insert(self):
        self.assertEqual(0., typo_distance("two", "two"))
        self.assertEqual(1., typo_distance("two", "twom"))
        self.assertEqual(2., typo_distance("two", "wtwom"))

    def test_substitute(self):
        self.assertEqual(0., typo_distance("two", "two"))
        self.assertEqual(2., typo_distance("two", "teo"))
        self.assertEqual(4., typo_distance("two", "teu"))

if __name__ == '__main__':
    unittest.main()
