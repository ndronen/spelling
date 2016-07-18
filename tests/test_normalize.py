import unittest
import spelling.normalize as normalize
import spacy

class TestNormalize(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.load('en')

    def test_digits(self):
        digits = {
                "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9
                }
        for text, num in digits.items():
            self.assertEqual(num, normalize.text2int(text))

    def test_scales(self):
        scales = {
                "one": 1, "ten": 10, "one hundred": 100, "one thousand": 1000
                }
        for text, num in scales.items():
            self.assertEqual(num, normalize.text2int(text))

    def test_hyphen(self):
        hyphens = {
                "twenty-two": 22, 
                "one hundred twenty-two": 122,
                "one hundred and twenty-two": 122
                }
        for text, num in hyphens.items():
            self.assertEqual(num, normalize.text2int(text))

    def test_is_hyphenated_compound(self):
        text = u'It might be forty-2, ten, sixty-two, twenty - five, or one hundred twenty-seven MPH.'
        doc = self.nlp(text)

        expectations = (
                (0, "It might", False),
                (4, "forty-2", False),
                (8, "sixty-two", True),
                (12, "twenty - five", False),
                (21, "MPH.", False)
                )

        for i,tokens,expected in expectations:
            self.assertEqual(expected, normalize.is_hyphenated_compound(doc, i))

    def test_group_number_word_tokens(self):
        text = u'There are one hundred and thirty-two elves.'
        doc = self.nlp(text)
        expected = [['There'], ['are'], ['one', 'hundred', 'and', 'thirty', '-', 'two'], ['elves'], ['.']]
        actual = normalize.group_number_word_tokens(doc)
        self.assertEqual(len(expected), len(actual))

    def test_words2digits(self):
        text = u'There are one hundred and thirty-two elves.'
        expected = u'There are 132 elves.'
        actual = normalize.words2digits(text)
        self.assertEqual(expected, actual)

if __name__ == '__main__':
    unittest.main()
