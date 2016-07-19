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
        self.expected_spans = (
                (0, 2, 3),
                (3, 7, 8),
                (8, 12, 13),
                (13, 16, 17),
                (17, 21, 21),
                (21, 22, 23),
                (23, 28, 29),
                (29, 36, 36),
                (36, 37, 37),
                (37, 40, 41),
                (41, 42, 43),
                (43, 49, 50),
                (50, 51, 52),
                (52, 55, 56),
                (56, 64, 64),
                (64, 65, 65)
                )

    def test_build_spans(self):
        text = u'is THAT what you mean, Capt. Donovan?for a moment i was confused.'
        tokenizer = Tokenizer()
        actual = tokenizer.build_spans(text)
        self.assertEqual(self.expected_spans, actual)

    def test_call(self):
        text = u'is THAT what you mean, Capt. Donovan?for a moment i was confused.'
        tokenizer = Tokenizer()
        tokens = tokenizer(text)
        self.assertEqual(len(self.expected_spans), len(tokens))
        self.assertEqual('is', tokens[0].text)
        self.assertEqual('is ', tokens[0].text_with_ws)
        self.assertEqual('confused', tokens[-2].text)
        self.assertEqual('confused', tokens[-2].text_with_ws)

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
                    'like_num': True,
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
                    'like_num': False,
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
                    self.assertEqual(expected, actual)
                except NotImplementedError:
                    print('%s is not implemented' % func)

if __name__ == '__main__':
    unittest.main()
