import unittest
from spelling.tokenize import SentenceSegmenter
from spelling.tokenize import Tokenizer
from spelling.tokenize import Token

class TestSentenceSegmenter(unittest.TestCase):
    def test_call(self):
        text = u'is THAT what you mean, Capt. Donovan?for a moment i was confused.'
        segmenter = SentenceSegmenter()
        self.assertEqual(2, len(segmenter(text)))
        segmenter = SentenceSegmenter(add_space_after_punctuation=False)
        self.assertEqual(1, len(segmenter(text)))

class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.text = u' is THAT what you mean, Capt. Donovan?for a moment i was confused.'
        self.tokenizer = Tokenizer()
        self.expected_spans = (
                (1, 3, 4),
                (4, 8, 9),
                (9, 13, 14),
                (14, 17, 18),
                (18, 22, 22),
                (22, 23, 24),
                (24, 29, 30),
                (30, 37, 37),
                (37, 38, 38),
                (38, 41, 42),
                (42, 43, 44),
                (44, 50, 51),
                (51, 52, 53),
                (53, 56, 57),
                (57, 65, 65),
                (65, 66, 66)
                )

    def test_next_space(self):
        i = self.tokenizer.next_space('x ')
        self.assertEqual(1, i)

    def test_next_non_space(self):
        i = self.tokenizer.next_non_space(' x')
        self.assertEqual(1, i)

    def test_find_new_position(self):
        text = u'A url http://www.example.com was used here.'

        tokens = self.tokenizer.tokenizer.tokenize(text)
        # ['A', 'url', 'http', ':', '//www.example.com', 'was', 'used', 'here', '.']
        self.assertTrue(tokens is not None)
        self.assertTrue(len(tokens) == 9)
        self.assertTrue(tokens[2] == 'http')

        # Set position to the index of 'http' in the token list.
        i = 2
        # Set character index to the 'w' of 'was'
        end_with_ws = text.index('was')

        new_i = self.tokenizer.find_new_position(text[end_with_ws:], tokens, i)
        self.assertEqual(tokens.index('was'), new_i)

    def test_handle_simple_token(self):
        tokens = self.tokenizer.tokenizer.tokenize(self.text)
        start = self.tokenizer.next_non_space(self.text)
        i, end, end_with_ws = self.tokenizer.handle_simple_token(
                self.text, tokens, i=0, start=start)
        self.assertEqual(1, i)
        self.assertEqual(3, end)
        self.assertEqual(4, end_with_ws)

    def test_handle_url_token(self):
        url = u'http://www.example.com'
        text = u'A url %s was used here.' % url

        tokens = self.tokenizer.tokenizer.tokenize(text)
        # ['A', 'url', 'http', ':', '//www.example.com', 'was', 'used', 'here', '.']
        self.assertTrue(tokens is not None)
        self.assertTrue(len(tokens) == 9)
        self.assertTrue(tokens[2] == 'http')

        i, end, end_with_ws = self.tokenizer.handle_url_token(
                text, tokens, i=2, start=6)

        self.assertEqual(tokens.index('was'), i)
        self.assertEqual(text.index(url) + len(url), end)
        self.assertEqual(text.index(url) + len(url) + 1, end_with_ws)

    def test_build_spans(self):
        actual = self.tokenizer.build_spans(self.text)
        self.assertEqual(self.expected_spans, actual)

    def test_call(self):
        tokens = self.tokenizer(self.text)
        self.assertEqual(len(self.expected_spans), len(tokens))
        self.assertEqual('is', tokens[0].text)
        self.assertEqual('is ', tokens[0].text_with_ws)
        self.assertEqual('confused', tokens[-2].text)
        self.assertEqual('confused', tokens[-2].text_with_ws)

    def test_inputs_that_break_tokenizer(self):
        texts = [u'well neva was seeing how many crickets the lizard would eat and she can see how many crickets to feed her lizard by boiling at the graph perhaps shes boiling on a trip, she could look at the graph and see"i need to put 60 crickets because i\'m boiling on a twelve day trip".']
        tokenizer = Tokenizer()
        for text in texts:
            tokenizer(text)

@unittest.skip('')
class TestToken(unittest.TestCase):
    def test_text_with_ws(self):
        text = u'This is an 8-word sentence with nine tokens.'
        expectations = (
                {
                    'text': 'This',
                    'text_with_ws': 'This ',
                    'whitespace': ' ',
                    'is_alpha': True, 
                    'is_ascii': True,
                    'is_digit': False,
                    'like_num': False,
                    'like_email': False,
                    'like_url': False
                    },
                {
                    'text': 'is',
                    'text_with_ws': 'is ',
                    'whitespace': ' ',
                    'is_alpha': True, 
                    'is_ascii': True,
                    'is_digit': False,
                    'like_num': False,
                    'like_email': False,
                    'like_url': False
                    },
                {
                    'text': 'an',
                    'text_with_ws': 'an ',
                    'whitespace': ' ',
                    'is_alpha': True, 
                    'is_ascii': True,
                    'is_digit': False,
                    'like_num': False,
                    'like_email': False,
                    'like_url': False
                    },
                {
                    'text': '8-word',
                    'text_with_ws': '8-word ',
                    'whitespace': ' ',
                    'is_alpha': False, 
                    'is_ascii': True,
                    'is_digit': False,
                    'like_num': False,
                    'like_email': False,
                    'like_url': False
                    },
                {
                    'text': 'sentence',
                    'text_with_ws': 'sentence ',
                    'whitespace': ' ',
                    'is_alpha': True, 
                    'is_ascii': True,
                    'is_digit': False,
                    'like_num': False,
                    'like_email': False,
                    'like_url': False
                    },
                {
                    'text': 'with',
                    'text_with_ws': 'with ',
                    'whitespace': ' ',
                    'is_alpha': True, 
                    'is_ascii': True,
                    'is_digit': False,
                    'like_num': False,
                    'like_email': False,
                    'like_url': False
                    },
                {
                    'text': 'nine',
                    'text_with_ws': 'nine ',
                    'whitespace': ' ',
                    'is_alpha': True, 
                    'is_ascii': True,
                    'is_digit': False,
                    'like_num': True,
                    'like_email': False,
                    'like_url': False
                    },
                {
                    'text': 'tokens',
                    'text_with_ws': 'tokens',
                    'whitespace': '',
                    'is_alpha': True, 
                    'is_ascii': True,
                    'is_digit': False,
                    'like_num': False,
                    'like_email': False,
                    'like_url': False
                    },
                {
                    'text': '.',
                    'text_with_ws': '.',
                    'whitespace': '',
                    'is_alpha': False, 
                    'is_ascii': True,
                    'is_digit': False,
                    'like_num': False,
                    'like_email': False,
                    'like_url': False
                    }
                )
        tokenizer = Tokenizer()
        tokens = tokenizer(text)
        for i,token in enumerate(tokens):
            for func, expected in expectations[i].items():
                try:
                    actual = getattr(token, func)
                    #print(token, func, expected, actual)
                    self.assertEqual(expected, actual)
                except NotImplementedError:
                    print('%s is not implemented' % func)

    def test_like_email(self):
        token = Token('user@example.com ', 0, 0, 15, 16)
        self.assertTrue(token.like_email)

    def test_like_url(self):
        token = Token('http://www.example.com ', 0, 0, 22, 23)
        self.assertTrue(token.like_url)


if __name__ == '__main__':
    unittest.main()
