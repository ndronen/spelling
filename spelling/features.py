from __future__ import absolute_import

import Levenshtein
import fuzzy
import string
from .typodistance import typo_distance

VOWELS = 'aeiou'
CONSONANTS = [l for l in string.ascii_letters if l not in VOWELS]

"""
Compute binary features between two words.

Parameters
----------
word1 : str
    The first of the two words.
word2 : str
    The second of the two words.

Returns
features : dict
    A dictionary with the keys 'levenshtein_distance',
    'keyboard_distance', 'soundex_levenshtein_distance',
    and 'metaphone_levenshtein_distance'.
"""
def compute_binary_features(word1, word2):
    features = {}
    features['levenshtein_distance'] = levenshtein_distance(word1, word2)
    features['keyboard_distance'] = keyboard_distance(word1, word2)
    features['soundex_levenshtein_distance'] = soundex_levenshtein_distance(word1, word2)
    features['metaphone_levenshtein_distance'] = metaphone_levenshtein_distance(word1, word2)
    return features

"""
Compute unary features of a word.

Parameters
----------
word : str
    The word for which to compute features.

Returns
features : dict
    A dictionary with the keys 'contains_space', 'character_count',
    'consonant_count', 'vowel_count', and 'capital_count'.
"""
def compute_unary_features(word):
    features = {}
    features['contains_space'] = contains_space(word)
    features['character_count'] = character_count(word)
    features['consonant_count'] = consonant_count(word)
    features['vowel_count'] = vowel_count(word)
    features['capital_count'] = capital_count(word)
    return features

"""
Compute levenshtein distance between two words.

Parameters
----------
word1 : str
    The first of the two words.
word2 : str
    The second of the two words.

Returns
distance : int
    The Levenshtein distance between `word1` and `word2`.
"""
def levenshtein_distance(word1, word2):
    return Levenshtein.distance(word1, word2)

"""
Parameters
----------
word1 : str
    The first of the two words.
word2  : str
    The second of the two words.

Returns
distance : float
    The keyboard distance between `word1` and `word2`.
"""
def keyboard_distance(word1, word2):
    return typo_distance(word1, word1)

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
    return f(word)[0]

"""
Levenshtein distance between SOUNDEX of two words.
"""
def soundex_levenshtein_distance(word1, word2, size=4):
    return levenshtein_distance(
            soundex(word1, size), soundex(word2, size))

"""
Levenshtein distance between Metaphone-2 of two words.
"""
def metaphone_levenshtein_distance(word1, word2):
    return levenshtein_distance(
            metaphone(word1)[0], metaphone(word2)[0])

"""
The dictionary suggestions.
"""
def suggest(dictionary, word):
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
