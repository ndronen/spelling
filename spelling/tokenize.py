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

class TokenizationError(Exception):
    pass

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
        return tokens

    def next_non_space(self, text):
        i = 0
        next_char = text[i]
        while next_char.isspace():
            i += 1
            next_char = text[i]
        return i

    def next_space(self, text):
        i = 0
        next_char = text[i]
        while not next_char.isspace():
            i += 1
            next_char = text[i]
        return i

    def find_new_position(self, text, tokens, i):
        """
        Parameters
        ----------
        text : str
            An untokenized string.
        tokens : list of str
            The list of tokens of `text`.
        i : int
            The index in `token` of the initial, old position.
        """
        new_i = i
        first_token_in_text = self.tokenizer.tokenize(text)[0]
        while tokens[new_i] != first_token_in_text:
            new_i += 1
        return new_i

    def handle_url_token(self, text, tokens, i, start):
        """
        Handle a URL token.

        Parameters
        ----------
        text : str
            The text being tokenized.
        tokens : list of str
            The list of tokens from `text`.
        i : int
            The index in `tokens` of the current token being processed.
        start : int
            The character index in `text` of the beginning of the current
            token `i` in `tokens`.

        Returns
        ----------
        i : int
            The index of the next token in `tokens` to be processed.  
        end : int
            The character index in `text` of the end of the token that
            was just processed.
        end_with_ws : int
            The character index in `text` of the end of the token that
            was just processed, including trailing whitespace.
        """
        token = tokens[i]

        if not token.startswith('http'):
            raise TokenizationError()

        # Take all characters from start to the next space.
        end = start + self.next_space(text[start:])

        # Then all characters from there to the next non-space.
        end_with_ws = end + self.next_non_space(text[end:])

        if end - start == len(token):
            # The entire URL was contained in the token, so we can simply
            # increment i.
            i = i + 1
        else:
            # We just consumed more than the token at position i, so
            # find the new position.
            i = self.find_new_position(text[end_with_ws:], tokens, i)

        return i, end, end_with_ws 

    def handle_simple_token(self, text, tokens, i, start):
        """
        Handle a simple token.

        Parameters
        ----------
        text : str
            The text being tokenized.
        tokens : list of str
            The list of tokens from `text`.
        i : int
            The index in `tokens` of the current token being processed.
        start : int
            The character index in `text` of the beginning of the current
            token `i` in `tokens`.

        Returns
        ----------
        i : int
            The index of the next token in `tokens` to be processed.  
        end : int
            The character index in `text` of the end of the token that
            was just processed.
        end_with_ws : int
            The character index in `text` of the end of the token that
            was just processed, including trailing whitespace.
        """
        end = start + len(tokens[i])

        if i+1 == len(tokens):
            end_with_ws = end
        else:
            end_with_ws = end + self.next_non_space(text[end:])

        return i+1, end, end_with_ws 

    def handle_token(self, text, tokens, i, start):
        if tokens[i].startswith('http'):
            try:
                return self.handle_url_token(text, tokens, i, start)
            except TokenizationError:
                return self.handle_simple_token(text, tokens, i, start)
        else:
            return self.handle_simple_token(text, tokens, i, start)

    def build_spans(self, text):
        i = idx = start = end = end_with_ws = 0
        if text[0].isspace():
            start = end = end_with_ws = self.next_non_space(text)
        tokens = self.tokenizer.tokenize(text)

        for j, t in enumerate(tokens):
            if t == '``':
                t = '"'
            elif t == "''":
                t = '"'
            tokens[j] = t

        spans = []
        while i < len(tokens):
            i, end, end_with_ws = self.handle_token(text, tokens, i, start)
            spans.append((start, end, end_with_ws))
            start = end_with_ws
        return tuple(spans)

class Token(object):
    TLDs = set("com|org|edu|gov|net|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|"
        "name|pro|tel|travel|xxx|"
        "ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|"
        "bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|"
        "co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|"
        "fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|"
        "hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|"
        "km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|"
        "mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|"
        "nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|"
        "sb|sc|sd|se|sg|sh|si|sj|sk|sl|sm|sn|so|sr|ss|st|su|sv|sy|sz|tc|td|tf|tg|th|"
        "tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|"
        "wf|ws|ye|yt|za|zm|zw".split('|'))

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
                'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety',
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
        """
        See https://github.com/spacy-io/spaCy/blob/master/spacy/orth.pyx
        """
        return self.like_email_(self.text) is not None

    @property
    def like_url(self):
        """
        See https://github.com/spacy-io/spaCy/blob/master/spacy/orth.pyx
        """
        if self.text.startswith('http://') or self.text.startswith('https://'):
            return True
        elif self.text.startswith('www.') and len(self.text) >= 5:
            return True
        if self.text[0] == '.' or self.text[-1] == '.':
            return False
        for i in range(len(self.text)):
            if self.text[i] == '.':
                break
        else:
            return False
        tld = self.text.rsplit('.', 1)[1].split(':', 1)[0]
        if tld.endswith('/'):
            return True
        if tld.isalpha() and tld in Token.TLDs:
            return True

        return False

    @property
    def whitespace(self):
        return self.text_with_ws[len(self.text):]

    def __repr__(self):
        return self.text
