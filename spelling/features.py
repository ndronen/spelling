import Levenshtein
import fuzzy
import string

VOWELS = 'aeiou'
CONSONANTS = [l for l in string.ascii_letters if l not in VOWELS]

"""
Compute levenshtein distance between two words.

Parameters
----------
w1 : str
    The first of the two words.
w2 : str
    The second of the two words.

Returns
distance : int
    The Levenshtein distance between `w1` and `w2`.
"""
def levenshtein(word1, word2):
    return Levenshtein.distance(word1, word2)

"""
Compute SOUNDEX of a word.
"""
def soundex(word, size=4):
    f = fuzzy.Soundex(size)
    return f(word)

"""
Compute Metaphone-2 of a word.
"""
def metaphone(word):
    f = fuzzy.DMetaphone()
    return f(word)

"""
Whether the SOUNDEX of two words are equal.
"""
def soundex_equal(word1, word2, size=4):
    return soundex(word1, size) == soundex(word2, size)

"""
Whether the Metaphone-2 of two words are equal.
"""
def metaphone_equal(word1, word2):
    return metaphone(word1) == metaphone(word2)

"""
The dictionary suggestions.
"""
def suggestions(dictionary, word):
    return dictionary.suggest(word)

"""
Whether a suggestion contains a space.
"""
def contains_space(word):
    return ' ' in word

"""
The length of a word.
"""
def character_count(word):
    return len(word)

"""
The number of consonants in a word.
"""
def consonant_count(word):
    word = str(word).lower()
    return len([c for c in word if c in CONSONANTS])

"""
The number of vowels in a word.
"""
def vowel_count(word):
    word = str(word)
    return len([c for c in word if c in VOWELS])

"""
The number of capital letters in a word.
"""
def capital_count(word):
    lower_word = word.lower()
    n_capitals = 0
    for i, c in enumerate(word):
        if c != lower_word[i]:
            n_capitals += 1
    return n_capitals
