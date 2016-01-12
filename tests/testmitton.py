# -*- coding: utf-8 -*-
import unittest
from spelling.mitton import *

class TestMitton(unittest.TestCase):
    def test_build_probs_dict(self):
        probs = build_probs_dict()
        self.assertEqual(0., probs["hakjahsdf8asdfa"])
        self.assertTrue(probs["hello"] > 0.)

if __name__ == '__main__':
    unittest.main()
