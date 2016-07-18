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

    def test_group_numeric_tokens(self):
        text = u'It might be ten, sixty-two, twenty five, or one hundred twenty-seven MPH.'
        doc = self.nlp(text)
        groups = normalize.group_numeric_tokens(doc)
        print(groups)
        #self.assertEqual(2, len(groups))
        #self.assertEqual(2, len(groups[0]))
        #self.assertEqual(4, len(groups[1]))

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
            try:
                self.assertEqual(expected, normalize.is_hyphenated_compound(doc, i))
            except AssertionError as e:
                print(i, tokens, expected, e)
                raise e

if __name__ == '__main__':
    unittest.main()
