import spacy

def build_numwords():
    units = [
        'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
        'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen',
        'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen',
        'nineteen',
    ]

    tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty',
            'seventy', 'eighty', 'ninety']

    scales = ['hundred', 'thousand', 'million', 'billion', 'trillion']

    numwords = { 'and': (1, 0) }

    for idx, word in enumerate(units):
        numwords[word] = (1, idx)
    for idx, word in enumerate(tens):
        numwords[word] = (1, idx * 10)
    for idx, word in enumerate(scales):
        numwords[word] = (10 ** (idx * 3 or 2), 0)

    return numwords

def text2int(textnum, numwords={}):
    if not numwords:
        numwords = build_numwords()

    ordinal_words = {
            'first': 1, 'second': 2, 'third': 3, 'fifth': 5,
            'eighth': 8, 'ninth': 9, 'twelfth': 12
            }
    ordinal_endings = [('ieth', 'y'), ('th', '')]
    textnum = textnum.replace('-', ' ')
    current = result = 0

    for word in textnum.split():
        if word in ordinal_words:
            scale, increment = (1, ordinal_words[word])
        else:
            for ending, replacement in ordinal_endings:
                if word.endswith(ending):
                    word = '%s%s' % (word[:-len(ending)], replacement)

            if word not in numwords:
                raise Exception('Illegal word: ' + word)

            scale, increment = numwords[word]

        current = current * scale + increment

        if scale > 100:
            result += current
            current = 0

    return result + current

def is_at_boundary(doc, i):
    return i == 0 or i+1 == len(doc)

def is_hyphenated_compound(doc, i):
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
    doc : spacy.token.doc.Doc
        A spaCy document.
    i : int
        The offset of a hyphen in the document.

    Returns
    ----------
    True if the hyphen is part of a hyphenated compound, False otherwise.
    """

    if is_at_boundary(doc, i):
        return False

    # Verify that the middle token is indeed a hyphen and that it is not
    # followed by whitespace.
    if doc[i].text_with_ws != '-':
        return False

    if not any([t.is_alpha for t in (doc[i-1], doc[i+1])]):
        return False

    # At this point in the function, we know that there is no spacing
    # after the hyphen, so all that remains is to check for spacing
    # after the preceding token.
    if doc[i-1].text != doc[i-1].text_with_ws:
        return False

    return True

def is_number_word(doc, i):
    return doc[i].is_alpha and doc[i].like_num

def is_immediate_context_number_words(doc, i):
    """
    Returns
    --------
    True if the tokens preceding and following the token at position i
    are number words, False otherwise.
    """
    return is_number_word(doc, i-1) and is_number_word(doc, i+1)

def is_hyphenated_number_word_compound(doc, i):
    if not is_hyphenated_compound(doc, i):
        return False

    return is_immediate_context_number_words(doc, i)

def is_number_word_conjunction(doc, i):
    """
    e.g. "hundred and twenty"
    """
    if is_at_boundary(doc, i):
        return False

    if doc[i].text.lower() != 'and':
        return False

    return is_immediate_context_number_words(doc, i)

def group_number_word_tokens(doc):
    """
    Aggregates number-word tokens (e.g. ``one hundred'', ``fifty-two'', 
    ``one hundred and fifty-two'').  Given the tokenized sentence

        ['There', 'are', 'one', 'hundred', 'and', 'thirty', '-', 'two', 'elves', '.'],

    this function builds a list of lists in which sequences of number-word
    tokens are in the same list, as in

        [['There'], ['are'], ['one', 'hundred', 'and', 'thirty', '-', 'two'], ['elves'], ['.']]

    Parameters
    ----------
    doc : spacy.token.doc.Doc
        A spaCy document.

    Returns
    ----------
    A list of list of tokens, with related, sequential number-words
    grouped together in the same sublist.
    """
    groups = []
    i = 0
    prev_numeric = -1
    while i < len(doc):
        token = doc[i]
        if token.text == '-':
            # If the preceding token, this hyphen, and the following
            # token comprise a hyphenated number-word compound, then
            # add the hyphen to the current group.  The following token
            # will automatically be added to the current group on the
            # next iteration.
            if is_hyphenated_number_word_compound(doc, i):
                groups[-1].append(token)
                prev_numeric = i
            else:
                groups.append([])
        elif token.text.lower() == 'and':
            # Group "... one hundred and twenty ... " into
            # [[...], ['one', 'hundred', 'and', 'twenty'], [...]].
            if is_number_word_conjunction(doc, i):
                groups[-1].append(token)
                prev_numeric = i
            else:
                groups.append([])
        else:
            # The token is either numeric or just a word.  If it's just
            # a word, we will append a new group.  If it is numeric, we
            # will not append a new group unless the previous token was
            # non-numeric.
            if not token.like_num or prev_numeric+1 < i:
                groups.append([])

            if token.like_num:
                prev_numeric = i

            groups[-1].append(token)

        i += 1

    return groups

try:
    nlp
except Exception:
    nlp = spacy.load('en')

def words2digits(text):
    doc = nlp(text)
    groups = group_number_word_tokens(doc)
    strings = []
    i = 0
    for group in groups:
        if len(group) == 1:
            token = group[0]
            if is_number_word(doc, i):
                ws = token.text_with_ws[len(token.text):]
                strings.append(text2int(token.text) + ws)
            else:
                strings.append(token.text_with_ws)
            i += 1
        else:
            # Aggregate the group into a single string, and convert the
            # aggregated string to a digit.
            strings.append(''.join([token.text_with_ws for token in group]))
            last_token = group[-1]
            last_ws = last_token.text_with_ws[len(last_token.text):]
            strings[-1] = str(text2int(strings[-1])) + last_ws
            i += len(group)

    return ''.join(strings)
