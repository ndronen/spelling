from __future__ import absolute_import

import Levenshtein
import fuzzy
import string
from .typodistance import typo_distance
import numpy as np

VOWELS = 'aeiou'
CONSONANTS = [l for l in string.ascii_letters if l not in VOWELS]

"""
Compute binary features between two words.

Parameters
----------
known_word : str
    The known word.
error : str
    The unknown word.

Returns
features : dict
    A dictionary with the keys 'levenshtein_distance',
    'keyboard_distance', 'soundex_levenshtein_distance',
    and 'metaphone_levenshtein_distance'.
"""
def compute_binary_features(known_word, unknown_word):
    features = {}
    features['levenshtein_distance'] = levenshtein_distance(known_word, unknown_word)
    features['keyboard_distance'] = keyboard_distance(known_word, unknown_word)
    features['set_distance'] = set_distance(known_word, unknown_word)
    features['soundex_levenshtein_distance'] = soundex_levenshtein_distance(known_word, unknown_word)
    features['metaphone_levenshtein_distance'] = metaphone_levenshtein_distance(known_word, unknown_word)
    features['soundex_set_distance'] = soundex_set_distance(known_word, unknown_word)
    features['metaphone_set_distance'] = metaphone_set_distance(known_word, unknown_word)

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
known_word : str
    The first of the two words.
unknown_word : str
    The second of the two words.

Returns
distance : int
    The Levenshtein distance between `known_word` and `unknown_word`.
"""
def levenshtein_distance(known_word, unknown_word):
    return Levenshtein.distance(known_word, unknown_word)

"""
Parameters
----------
known_word : str
    The first of the two words.
unknown_word  : str
    The second of the two words.

Returns
distance : float
    The keyboard distance between `known_word` and `unknown_word`.
"""
def keyboard_distance(known_word, unknown_word):
    return typo_distance(known_word, unknown_word)

"""
Compute SOUNDEX of a word.
"""
def soundex(word, size=4):
    f = fuzzy.Soundex(size)
    sx = f(word)
    if sx is None:
        sx == ''
    return sx

"""
Compute Metaphone-2 of a word.
"""
def metaphone(word):
    f = fuzzy.DMetaphone()
    mph = f(word)[0]
    if mph is None:
        mph = ''
    return mph

"""
Levenshtein distance between SOUNDEX of two words.
"""
def soundex_levenshtein_distance(known_word, unknown_word, size=4):
    try:
        return levenshtein_distance(
                soundex(known_word, size), soundex(unknown_word, size))
    except IndexError:
        if len(known_word) == 0 and len(unknown_word) == 0:
            return np.inf
        else:
            return max(len(known_word), len(unknown_word))

"""
Levenshtein distance between Metaphone-2 of two words.
"""
def metaphone_levenshtein_distance(known_word, unknown_word):
    try:
        return levenshtein_distance(
                metaphone(known_word), metaphone(unknown_word))
    except IndexError:
        if len(known_word) == 0 and len(unknown_word) == 0:
            return np.inf
        else:
            return max(len(known_word), len(unknown_word))

"""
Unordered distance of two words.  (This is probably a proper metric.)
"""
def set_distance(known_word, unknown_word):
    s_known = set(known_word)
    s_unknown = set(unknown_word)
    try:
        return 1 - len(s_unknown.intersection(s_known))/float(len(s_unknown))
    except ZeroDivisionError:
        return len(s_known)

"""
Unordered distance of SOUNDEX of two words.  (This is probably a proper metric.)
"""
def soundex_set_distance(known_word, unknown_word):
    return set_distance(soundex(known_word), soundex(unknown_word))

"""
Unordered distance of Metaphone of two words.  (This is probably a proper metric.)
"""
def metaphone_set_distance(known_word, unknown_word):
    return set_distance(metaphone(known_word), metaphone(unknown_word))

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
    word = word.lower()
    return len([c for c in word if c in CONSONANTS])

"""
The number of vowels in a word.
"""
def vowel_count(word):
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
