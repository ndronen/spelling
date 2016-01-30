import unittest
import spelling.mitton
from spelling.edits import EditFinder, Editor

class TestEditor(unittest.TestCase):
    def setUp(self):
        self.editor = Editor()

    def test_insert(self):
        edits = self.editor.insert("food")
        self.assertTrue('fozod' in edits)
        edits = self.editor.edit("food", "insert")
        self.assertTrue('fozod' in edits)

    def test_delete(self):
        edits = self.editor.delete("food")
        self.assertTrue('fod' in edits)

    def test_substitute(self):
        edits = self.editor.substitute("food")
        self.assertTrue('zood' in edits)

    def test_transpose(self):
        edits = self.editor.transpose("food")
        self.assertTrue('ofod' in edits)

    def test_split(self):
        edits = self.editor.split("food")
        self.assertTrue('fo od' in edits)

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
        print word, error, edits
        self.assertEquals([('ar', 'ax')], edits)

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
                    'expected': ('th', 'thr')
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
        expected = ("ar", "ax")
        first, second = self.finder.align(word, error)
        # The words are aligned like this:
        #     "scar"
        #     "scax"
        start = 3
        end = start
        self.assertTrue(self.finder.edit_is_substitution(first, second, start, end))
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
        expected = [('ac','as'), ('ce', 'c'), ('nt', 'nts')]
        edits = self.finder.build_edits(first, second)
        self.assertEquals(expected, edits)

    #@unittest.skip('')
    def test_apply_straight(self):
        word =  "straight"
        error = "strait"
        edits = self.finder.find(word, error)
        self.assertEquals(error, self.finder.apply(word, edits))

    #@unittest.skip('')
    def test_apply_generally(self):
        word =  "generally"
        error = "geneology"
        edits = self.finder.find(word, error)
        self.assertEquals(error, self.finder.apply(word, edits))

    #@unittest.skip('')
    def test_apply_critics(self):
        word =  "critics"
        error = "criticists"
        edits = self.finder.find(word, error)
        self.assertEquals(error, self.finder.apply(word, edits))

    #@unittest.skip('')
    def test_apply_professor(self):
        word =  "professor"
        error = "proffesor"
        edits = self.finder.find(word, error)
        self.assertEquals(error, self.finder.apply(word, edits))

    #@unittest.skip('')
    def test_apply_one(self):
        word =  "one"
        error = "noone"
        edits = self.finder.find(word, error)
        self.assertEquals(error, self.finder.apply(word, edits))

    #@unittest.skip('')
    def test_apply_throughout(self):
        word =  "throughout"
        error = "throught"
        edits = self.finder.find(word, error)
        self.assertEquals(error, self.finder.apply(word, edits))

    #@unittest.skip('')
    def test_remove_dashes(self):
        word =  "crit-cs"
        self.assertEquals("critcs", self.finder.remove_dashes(5, word)[1])
        self.assertEquals(4, self.finder.remove_dashes(5, word)[0])
        self.assertEquals(2, self.finder.remove_dashes(2, word)[0])
        self.assertEquals(4, self.finder.remove_dashes(4, word)[0])

    #@unittest.skip('')
    def test_remove_double_dashes(self):
        word =  "cr-t-cs"
        self.assertEquals("crtcs", self.finder.remove_dashes(5, word)[1])
        self.assertEquals(3, self.finder.remove_dashes(5, word)[0])
        self.assertEquals(1, self.finder.remove_dashes(1, word)[0])
        self.assertEquals(3, self.finder.remove_dashes(4, word)[0])

    #@unittest.skip('')
    def test_remove_no_dashes(self):
        word =  "critics"
        self.assertEquals("critics", self.finder.remove_dashes(5, word)[1])
        self.assertEquals(5, self.finder.remove_dashes(5, word)[0])

    #@unittest.skip('')
    def test_apply_on_wiki(self):
        words = spelling.mitton.load_mitton_words('data/wikipedia.dat')
        pairs = spelling.mitton.build_mitton_pairs(words)
        #with open("data/wikipedia.dat","r") as f:
        #    for line in f:
        #        if line[0] == "$":
        #            correct = line[1:-1]
        #        else:
        #            incorrect = line[:-1]
        for incorrect,correct in pairs:
            edits = self.finder.find(correct, incorrect)
            try:
                recovered_error = self.finder.apply(correct, edits)
                self.assertEquals(incorrect, recovered_error)
            except AssertionError as e:
                print(incorrect, correct, edits, recovered_error, e)
