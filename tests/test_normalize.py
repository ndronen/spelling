import unittest
import time
import spelling.normalize as normalize
from spelling.tokenize import Tokenizer
import spacy

class TestNormalize(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()
        """
        This test should pass using the spacy tokenizer, too.
        self.tokenizer = spacy.load('en',
            parser=False, tagger=False, entity=False, matcher=False, serializer=False)
        """

    def test_convert_integer_words_to_integers_function(self):
        digits = {
                "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9
                }
        for text, num in digits.items():
            self.assertEqual(num, normalize.convert_integer_words_to_integers(text))

    def test_scales(self):
        scales = {
                "one": 1, "ten": 10, "one hundred": 100, "one thousand": 1000
                }
        for text, num in scales.items():
            self.assertEqual(num, normalize.convert_integer_words_to_integers(text))

    def test_hyphen(self):
        hyphens = {
                "twenty-two": 22, 
                "one hundred twenty-two": 122,
                "one hundred and twenty-two": 122
                }
        for text, num in hyphens.items():
            self.assertEqual(num, normalize.convert_integer_words_to_integers(text))

    def test_is_hyphenated_compound(self):
        text = u'It might be forty-2, ten, sixty-two, twenty - five, or one hundred twenty-seven MPH.'
        tokens = self.tokenizer(text)

        if isinstance(self.tokenizer, Tokenizer):
            expectations = (
                    (0, "It might", False),
                    (4, "forty-2", False),
                    (7, "sixty-two", True),
                    (10, "twenty - five", False),
                    (17, "MPH", False)
                    )
        else:
            expectations = (
                    (0, "It might", False),
                    (4, "forty-2", False),
                    (8, "sixty-two", True),
                    (12, "twenty - five", False),
                    (21, "MPH.", False)
                    )

        for i,strings,expected in expectations:
            self.assertEqual(expected, normalize.is_hyphenated_compound(tokens, i))

    def test_group_integer_word_tokens(self):
        text = u'There are one hundred and thirty-two elves.'
        tokens = self.tokenizer(text)
        if isinstance(self.tokenizer, Tokenizer):
            expected = [['There'], ['are'], ['one', 'hundred', 'and', 'thirty-two'], ['elves'], ['.']]
        else:
            expected = [['There'], ['are'], ['one', 'hundred', 'and', 'thirty', '-', 'two'], ['elves'], ['.']]
        actual = normalize.group_integer_word_tokens(tokens)
        self.assertEqual(len(expected), len(actual))
        for i,group in enumerate(actual):
            self.assertEqual(len(expected[i]), len(actual[i]))

    def test_is_integer_word_conjunction(self):
        text = u'one hundred and thirty-two'
        tokens = self.tokenizer(text)
        actual = normalize.is_integer_word_conjunction(tokens, 2) # 'and'
        self.assertEqual(True, actual)

    def test_convert_integer_words_to_integers(self):
        expectations = (
                (u'There are one hundred and thirty-two elves.', u'There are 132 elves.'),
                (u'There are one hundred and thirty two elves.', u'There are 132 elves.'),
                (u'Five million five thousand one hundred and sixty-two.', u'5005162.'),
                (u'Five million five thousand one hundred sixty-two.', u'5005162.'),
                )
        transformer = normalize.ConvertIntegerWordsToIntegers(self.tokenizer)
        actual = transformer.transform([e[0] for e in expectations])
        for i,act in enumerate(actual):
            self.assertEqual(expectations[i][1], act)

if __name__ == '__main__':
    unittest.main()
