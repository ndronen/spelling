import re
import nltk.tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize.punkt import PunktParameters
    
# TODO: handle URLs.
class SentenceSegmenter(object):
    def __init__(self, add_space_after_punctuation=True, abbrev_types=None):
        self.add_space_after_punctuation = add_space_after_punctuation

        if abbrev_types is None:
            # TODO: expand this as needed.
            abbrev_types = set([
                'dr', 'vs', 'mr', 'mrs', 'prof',
                'col', 'capt', 'cpl', 'cmdr', 'comdr', 
                'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sept', 'oct', 'nov', 'dec',
                'inc', 'corp', 'dept', 'dist', 'div', 'ed', 'est',
                'ave', 'blvd', 'ln', 'rd', 'st'])

        self.splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.splitter._params.abbrev_types.update(abbrev_types)

    def __call__(self, text):
        # Force the text to have a space after every sentence-terminating
        # or -delimiting punctuation character.  Students can be lax in
        # their use of spacing and punctuation, so this gives us a chance
        # of recovering an intended segmentation.
        if self.add_space_after_punctuation:
            text = re.sub(r'([.?!;:])(\w)', r'\1 \2', text)

        return self.splitter.tokenize(text)

# TODO: handle URLs.
class Tokenizer(object):
    def __init__(self):
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def __call__(self, text):
        spans = self.build_spans(text)
        tokens = []
        for i, (start, end, end_with_ws) in enumerate(spans):
            token_text = text[start:end]
            token_text_with_ws = text[start:end_with_ws]
            token = Token(text, i, start, end, end_with_ws)
            tokens.append(token)
            #print("text '%s' token text '%s' token text with whitespace '%s'" % (
            #    text, token_text, token_text_with_ws))
        return tokens

    def next_non_space(self, text):
        i = 0
        next_char = text[i]
        while next_char.isspace():
            i += 1
            next_char = text[i]
        return i

    def build_spans(self, text):
        tokens = self.tokenizer.tokenize(text)
        spans = []
        idx = start = end = end_with_ws = 0
        for i,token in enumerate(tokens):
            start += self.next_non_space(text[start:])
            end = start + len(token)
            if i+1 == len(tokens):
                end_with_ws = end
            else:
                end_with_ws = end + self.next_non_space(text[end:])
            #print("i %d text '%s' token '%s' start %d end %d end_with_ws %d token_text '%s' token_text_with_ws '%s'" %
            #        (i, text, token, start, end, end_with_ws, text[start:end], text[start:end_with_ws]))
            spans.append((start, end, end_with_ws))

            start = end_with_ws
        return tuple(spans)

class Token(object):
    def __init__(self, doc, i, start, end, end_with_ws):
        self.i = i 
        self.idx = start
        self.start = start
        self.end = end
        self.end_with_ws = end_with_ws
        self.text = doc[self.start:self.end]
        self.text_with_ws = doc[self.start:self.end_with_ws]
        self.nums = set([
                'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
                'hundred', 'thousand', 'million', 'billion', 'trillion', 'quadrillion',
                'quintillion', 'sextillion', 'septillion', 'octillion', 'nonillion',
                'decillion', 'undecillion', 'duodecillion', 'tredecillion', 
                'quatttuor', 'quatttuor-decillion', 'quindecillion', 'sexdecillion',
                'septen', 'septen-decillion', 'octodecillion', 'novemdecillion',
                'vigintillion', 'centillion'
                ])
        self.like_email_ = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)").match

    @property
    def is_alpha(self):
        return self.text.isalpha()

    @property
    def is_ascii(self):
        return not any(ord(c) >= 128 for c in self.text)

    @property
    def is_digit(self):
        return self.text.isdigit()

    @property
    def like_num(self):
        if self.text.isdecimal():
            return True
        return self.text.lower() in self.nums

    @property
    def like_email(self):
        return self.like_email_(self.text) is not None

    @property
    def like_url(self):
        # TODO
        raise NotImplementedError()

    @property
    def whitespace(self):
        return self.text_with_ws[len(self.text):]

    def __str__(self):
        return self.text

    def __repr__(self):
        return str([self.text, self.start, self.end])
