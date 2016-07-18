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

    if i == 0:
        return False

    if i+1 == len(doc):
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

def group_numeric_tokens(doc):
    """
    """
    groups = [[]]
    i = 0
    while True:
        break
        """
        token = doc[i]
        if token.text == '-':
            # If there's no space between token i-1 and this token,
            # and between this token an token i+1, and if tokens i-1
            # and i+1 are number-like, then place the three tokens in
            # a single group.
            pass
        print(i, token)
        if token.like_num:
            groups[-1].append(X)
        """

"""
def group_adjacent_tokens(doc):
    groups = [[]]
    for i,token in enumerate(doc[:-1]):
        print(i, token)
        groups[-1].append(token)
        if doc[i+1].i - token.i > 1:
            # These two tokens are not adjacent in the sentence.
            groups.append([])
    groups[-1].append(doc[-1])
    return groups
"""

def find_misspelled_words(doc):
    """
    Parameters
    ---------
    doc : spacy.tokens.doc.Doc
        A sentence represented by spaCy.

    Returns
    ---------
    List of spacy.tokens.token.Token.
    """
    assert len([s for s in doc.sents]) == 1

def find_number_tokens(doc):
    """
    Parameters
    ---------
    doc : spacy.tokens.doc.Doc
        A sentence represented by spaCy.

    Returns
    ---------
    List of spacy.tokens.token.Token.
    """
    assert len([s for s in doc.sents]) == 1
    return [t for t in doc if t.like_num and t.is_alpha]

"""
def find_number_tokens(tokens, numwords={}):
    if not numwords:
        numwords = build_numwords()

    indices = []
    last_successful = -1
    print('tokens', tokens)
    print('numwords', list(numwords.keys()))
    for i,token in enumerate(tokens):
        print(i, token)
        try:
            numwords[token.lower()]
            indices.append(i)
        except KeyError:
            pass

    print('indices', indices)

    groups = [[]]
    for i,index in enumerate(indices[0:-1]):
        print(i, index)
        if indices[i+1] - index > 1:
            # Break in contiguity.
            groups.append([])
        groups[-1].append(tokens[index])
    return groups
"""

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
