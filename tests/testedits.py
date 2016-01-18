import unittest
from spelling.edits import EditFinder, TooManyEditsError

class TestEditFinder(unittest.TestCase):
    def setUp(self):
        self.finder = EditFinder()
    
    def test_words_with_different_length(self):
        a = "threw"
        b = "thew"
        edits = self.finder.find(a, b)
        self.assertEquals([('r', '-')], edits)
        edits = self.finder.find(b, a)
        self.assertEquals([('-', 'r')], edits)

    def test_too_many_edits(self):
        a = "car"
        b = "scare"
        self.assertRaises(TooManyEditsError,
                self.finder.find, a, b)
