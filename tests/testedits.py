import unittest
from spelling.edits import EditFinder

class TestEditFinder(unittest.TestCase):
    def setUp(self):
        self.finder = EditFinder()
    
    #@unittest.skip('')
    def test_deletion(self):
        word = "throne"
        error = "thron"
        edits = self.finder.find(word, error)
        self.assertEquals([('ne', 'n')], edits)

    #@unittest.skip('')
    def test_transposition(self):
        word = "their"
        error = "thier"
        edits = self.finder.find(word, error)
        self.assertEquals([('ei', 'ie')], edits)

    #@unittest.skip('')
    def test_substitution(self):
        word = "scar"
        error = "scax"
        edits = self.finder.find(word, error)
        self.assertEquals([('r', 'x')], edits)

    #@unittest.skip('')
    def test_build_edits_rotation(self):
        word = "tragedy"
        error = "tradegy"
        first, second = self.finder.align(word, error)
        start = 3
        end = start + 2
        self.assertTrue(self.finder.edit_is_rotation(first, second, start, end))
        edits = self.finder.build_edits(first, second)
        expected = [('aged', 'adeg')]
        self.assertEquals(expected, edits)

    #@unittest.skip('')
    def test_build_edits_transposition(self):
        word = "their"
        error = "thier"
        first, second = self.finder.align(word, error)
        # The words are aligned like this:
        #     "th-eir"
        #     "thei-r"
        # So a transposition spans three characters.
        start = 2
        end = start + 2
        self.assertTrue(self.finder.edit_is_transposition(first, second, start, end))
        expected = [('ei', 'ie')]
        edits = self.finder.build_edits(first, second)
        self.assertEquals(expected, edits)

    #@unittest.skip('')
    def test_build_edits_insertion(self):
        tests = [{
                    'word': 'the',
                    'error': 'thre',
                    'start': 2,
                    'expected': ('h', 'hr')
                },
                {
                    'word': 'car',
                    'error': 'pcar',
                    'start': 0,
                    'expected': ('^', '^p')
                }]
        for test in tests:
            word = test['word']
            error = test['error']
            start = test['start']
            end = start
            expected = test['expected']

            first, second = self.finder.align(word, error)
            self.assertTrue(self.finder.edit_is_insertion(first, second, start, end))
            edits = self.finder.build_insertion(first, second, start, end)
            self.assertEquals(expected, edits)
            edits = self.finder.build_edits(first, second)
            self.assertEquals([expected], edits)

    #@unittest.skip('')
    def test_build_edits_deletion(self):
        tests = [{
                    'word': 'three',
                    'error': 'thre',
                    'start': 4,
                    'expected': ('ee', 'e')
                },
                {
                    'word': 'three',
                    'error': 'hree',
                    'start': 0,
                    'expected': ('^t', '^')
                }]
        for test in tests:
            word = test['word']
            error = test['error']
            start = test['start']
            end = start
            expected = test['expected']
            first, second = self.finder.align(word, error)
            self.assertTrue(self.finder.edit_is_deletion(first, second, start, end))
            edits = self.finder.build_deletion(first, second, start, end)
            self.assertEquals(expected, edits)
            edits = self.finder.build_edits(first, second)
            self.assertEquals([expected], edits)

    #@unittest.skip('')
    def test_build_edits_substitution(self):
        word = "scar"
        error = "scax"
        expected = ("r", "x")
        first, second = self.finder.align(word, error)
        # The words are aligned like this:
        #     "scar"
        #     "scax"
        start = 3
        end = start
        self.assertTrue(self.finder.edit_is_substitution(first, second, start, end))
        expected = ('r', 'x')
        edits = self.finder.build_substitution(first, second, start, end)
        self.assertEquals(expected, edits)
        edits = self.finder.build_edits(first, second)
        self.assertEquals([expected], edits)

    #@unittest.skip('')
    def test_no_edits(self):
        word =  "replacement"
        error = "replasments"
        #
        # The words are aligned like this:
        #     "replacement-"
        #     "replas-ments"
        # This should be a substitution, a deletion, and an insertion.
        #     ('c','s'), ('ce', 'c'), ('t', 'ts')
        first, second = self.finder.align(word, error)
        expected = [('c','s'), ('ce', 'c'), ('t', 'ts')]
        edits = self.finder.build_edits(first, second)
        self.assertEquals(expected, edits)

    def test_apply_straight(self):
        word =  "straight"
        error = "strait"
        edits,_ = self.finder.find(word, error)
        self.assertEquals(error, self.finder.apply(word, edits))

    def test_apply_generally(self):
        word =  "generally"
        error = "geneology"
        edits,_ = self.finder.find(word, error)
        self.assertEquals(error, self.finder.apply(word, edits))

    def test_apply_critics(self):
        word =  "critics"
        error = "criticists"
        edits,_ = self.finder.find(word, error)
        self.assertEquals(error, self.finder.apply(word, edits))

    @unittest.skip('')
    def test_apply_on_wiki(self):
        with open("data/wikipedia.dat","r") as f:
            for line in f:
                if line[0] == "$":
                    correct = line[1:-1]
                else:
                    incorrect = line[:-1]
                edits,_ = self.finder.find(correct, incorrect)
                self.assertEquals(incorrect, self.finder.apply(correct, edits))
