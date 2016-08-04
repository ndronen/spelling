import re
import sklearn.base as skbase
import logging
from spelling.tokenize import Token

def build_integer_words():
    units = [
        'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
        'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen',
        'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen',
        'nineteen',
    ]

    tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty',
            'seventy', 'eighty', 'ninety']

    scales = ['hundred', 'thousand', 'million', 'billion', 'trillion']

    integer_words = { 'and': (1, 0) }

    for idx, word in enumerate(units):
        integer_words[word] = (1, idx)
    for idx, word in enumerate(tens):
        integer_words[word] = (1, idx * 10)
    for idx, word in enumerate(scales):
        integer_words[word] = (10 ** (idx * 3 or 2), 0)

    return integer_words

def convert_integer_words_to_integers(text, integer_words={}):
    if not integer_words:
        integer_words = build_integer_words()

    ordinal_words = {
            'first': 1, 'second': 2, 'third': 3, 'fifth': 5,
            'eighth': 8, 'ninth': 9, 'twelfth': 12
            }
    ordinal_endings = [('ieth', 'y'), ('th', '')]
    text = text.replace('-', ' ')
    current = result = 0

    for word in text.split():
        word = word.lower()
        if word in ordinal_words:
            scale, increment = (1, ordinal_words[word])
        else:
            for ending, replacement in ordinal_endings:
                if word.endswith(ending):
                    word = '%s%s' % (word[:-len(ending)], replacement)

            if word not in integer_words:
                raise Exception('Illegal word: ' + word)

            scale, increment = integer_words[word]

        current = current * scale + increment

        if scale > 100:
            result += current
            current = 0

    return result + current

def is_at_boundary(tokens, i):
    return i == 0 or i+1 == len(tokens)

hyphenated_compound = re.compile(r'(\w+)-(\w+)')

def is_hyphenated_compound(tokens, i):
    """
    Identify hyphenated compounds, such as ``twenty-six''.  Longer
    compounds, such as ``one-hundred-twenty-six'', can be assembled by
    repeated invocation of this function.
    
    Two requirements must be met for a pair of tokens on each side
    of a '-' to be considered a hyphenated compound.  Both tokens must
    consist only of alphabetical characters.  This constraint prevents
    numerical ranges (e.g. ``10-12'') from triggering a false positive.
    (The spaCy tokenizer already considers compounds like ``Catch-22''
    a single token, because only one of the tokens is a digit.)
    There must also be no spacing on either side of the hyphen.  This
    prevents false positives due to ranges (e.g. ``July - October 2010'').

    Parameters
    ----------
    tokens : A list of tokens.
        Each element can be an instance of spacy.tokens.token.Token
        or of an in-house Token class that has a similar interface.
    i : int
        The offset of a hyphen in the list of tokens.

    Returns
    ----------
    True if the hyphen is part of a hyphenated compound, False otherwise.
    """
    if re.match(hyphenated_compound, tokens[i].text) is not None:
        return True

    if is_at_boundary(tokens, i):
        return False

    # Verify that the middle token is indeed a hyphen and that it is not
    # followed by whitespace.
    if tokens[i].text_with_ws != '-':
        return False

    if not any([t.is_alpha for t in (tokens[i-1], tokens[i+1])]):
        return False

    # At this point in the function, we know that there is no spacing
    # after the hyphen, so all that remains is to check for spacing
    # after the preceding token.
    if tokens[i-1].text != tokens[i-1].text_with_ws:
        return False

    return True

def is_integer_word(tokens, i):
    def is_integer_word_(token):
        return token.is_alpha and token.like_num

    if is_integer_word_(tokens[i]):
        return True

    match = re.match(hyphenated_compound, tokens[i].text)
    if match is None:
        return False

    left_group = match.group(1)
    left_token = Token(left_group, 0, 0, len(left_group), len(left_group))

    right_group = match.group(2)
    right_token = Token(right_group, 0, 0, len(right_group), len(right_group))

    return is_integer_word_(left_token) and is_integer_word_(right_token)

def is_immediate_context_integer_words(tokens, i):
    """
    Returns
    --------
    True if the tokens preceding and following the token at position i
    are integer words, False otherwise.
    """
    return is_integer_word(tokens, i-1) and is_integer_word(tokens, i+1)

def is_hyphenated_integer_word_compound(tokens, i):
    if not is_hyphenated_compound(tokens, i):
        return False

    return is_immediate_context_integer_words(tokens, i)

def is_integer_word_conjunction(tokens, i):
    """
    e.g. "hundred and twenty"
    """
    if is_at_boundary(tokens, i):
        return False

    if tokens[i].text.lower() != 'and':
        return False

    return is_immediate_context_integer_words(tokens, i)

def group_integer_word_tokens(tokens):
    """
    Aggregates number-word tokens (e.g. ``one hundred'', ``fifty-two'', 
    ``one hundred and fifty-two'').  Given the tokenized sentence

        ['There', 'are', 'one', 'hundred', 'and', 'thirty', '-', 'two', 'elves', '.'],

    this function builds a list of lists in which sequences of number-word
    tokens are in the same list, as in

        [['There'], ['are'], ['one', 'hundred', 'and', 'thirty', '-', 'two'], ['elves'], ['.']]

    Parameters
    ----------
    tokens : A list of tokens.
        Each element can be an instance of spacy.tokens.token.Token
        or of an in-house Token class that has a similar interface.

    Returns
    ----------
    A list of list of tokens, with related, sequential number-words
    grouped together in the same sublist.
    """
    groups = []
    i = 0
    prev_numeric = -1
    while i < len(tokens):
        token = tokens[i]
        if token.text == '-':
            # If the preceding token, this hyphen, and the following
            # token comprise a hyphenated number-word compound, then
            # add the hyphen to the current group.  The following token
            # will automatically be added to the current group on the
            # next iteration.
            if is_hyphenated_integer_word_compound(tokens, i):
                groups[-1].append(token)
                prev_numeric = i
            else:
                groups.append([token])
        elif token.text.lower() == 'and':
            # Group "... one hundred and twenty ... " into
            # [[...], ['one', 'hundred', 'and', 'twenty'], [...]].
            if is_integer_word_conjunction(tokens, i):
                groups[-1].append(token)
                prev_numeric = i
            else:
                groups.append([token])
        else:
            # The token is either numeric or just a word.  If it's just
            # a word, we will append a new group.  If it is numeric, we
            # will not append a new group unless the previous token was
            # non-numeric.
            if not is_integer_word(tokens, i) or prev_numeric+1 < i:
                groups.append([])

            if is_integer_word(tokens, i):
                prev_numeric = i

            if len(groups) == 0:
                # First group.
                groups.append([])

            groups[-1].append(token)

        i += 1

    return groups

class ConvertIntegerWordsToIntegers(skbase.BaseEstimator,skbase.TransformerMixin):
    def __init__(self, tokenizer):
        """
        Parameters
        ----------
        tokenizer : a callable tokenizer object 
            This can be an instance of spacy.en.English or a Tokenizer.
            Both implementations return an iterable over tokens.
        """
        self.tokenizer = tokenizer

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        xformed = []
        for text in X:
            tokens = self.tokenizer(text)
            groups = group_integer_word_tokens(tokens)
            strings = []
            i = 0
            #print('text', text)
            #print('groups', groups)
            for group in groups:
                #print('group', group)
                if len(group) == 1:
                    token = group[0]
                    if is_integer_word(tokens, i):
                        ws = token.text_with_ws[len(token.text):]
                        strings.append(str(convert_integer_words_to_integers(token.text)) + ws)
                    else:
                        strings.append(token.text_with_ws)
                    i += 1
                else:
                    # Aggregate the group into a single string, and convert the
                    # aggregated string to a digit.
                    strings.append(''.join([token.text_with_ws for token in group]))
                    last_token = group[-1]
                    last_ws = last_token.text_with_ws[len(last_token.text):]
                    integer = convert_integer_words_to_integers(strings[-1])
                    strings[-1] = str(integer) + last_ws
                    i += len(group)
            xformed.append(''.join(strings))
        return xformed

class ConvertIntegersToRangeLabels(skbase.BaseEstimator,skbase.TransformerMixin):
    def __init__(self, tokenizer, min_range, max_range):
        """
        Replace integer tokens matching r'\W\d+\W' with either `INSIDERANGE`
        or `OUTSIDERANGE`, depending on whether the token x satisfies
        the inequality `min_range <= x <= max_range`.

        Parameters
        ----------
        tokenizer : a callable tokenizer object 
            This can be an instance of spacy.en.English or a Tokenizer.
            Both implementations return an iterable over tokens.
        min_range : int
            The minimum value of the range.
        max_range : int
            The maximum value of the range.
        """
        self.tokenizer = tokenizer
        self.min_range = min_range
        self.max_range = max_range

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        xformed = []
        for text in X:
            tokens = self.tokenizer(text)
            strings = []
            print('text', text)
            for i,token in enumerate(tokens):
                if token.is_digit:
                    print(token.text + ' is digit')
                    if self.min_range <= float(token.text) <= self.max_range:
                        strings.append('INSIDERANGE' + token.whitespace)
                    else:
                        strings.append('OUTSIDERANGE' + token.whitespace)
                else:
                    print(token.text + ' is not digit')
                    strings.append(token.text_with_ws)
            xformed.append(''.join(strings))
        return xformed
