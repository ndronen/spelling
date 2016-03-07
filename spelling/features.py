from __future__ import absolute_import

import sys

from jellyfish import soundex, metaphone, jaro_winkler, nysiis
from jellyfish import (levenshtein_distance, damerau_levenshtein_distance,
    hamming_distance, match_rating_comparison, jaro_distance)
import string
from .typodistance import typo_distance
import numpy as np

VOWELS = 'aeiou'
CONSONANTS = [l for l in string.ascii_letters if l not in VOWELS]
METRICS = ['levenshtein', 'damerau_levenshtein', 'hamming', 'jaro', 'jaro_winkler', 'typo', 'set']
ENCODINGS = ['identity', 'soundex', 'metaphone', 'nysiis']

if sys.version_info == 3:
    print('aliasing unicode to str')
    unicode = str
else:
    print('using python 2, leaving unicode function alone')
#unicode = unicode if 'unicode' in globals() else str

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
    A dictionary with each key being a feature.
"""
def compute_binary_features(known_word, unknown_word):
    features = {}
    for metric in METRICS:
        for encoding in ENCODINGS:
            feature_name = '_'.join([encoding, metric])
            features[feature_name] = distance(known_word, unknown_word,
                    metric, encoding=encoding)
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
Compute distance between two words (optionally, after encoding the words).

Parameters
----------
known_word : str
    The known word.
unknown_word : str
    The unknown word.
metric : str or callable
    The metric.
encoding : str or callable
    The encoding (default is identity).

Returns
distance : int
    The distance between `known_word` and `unknown_word`.
"""
def distance(known_word, unknown_word, metric, encoding=lambda s: s, verbose=False):
    try:
        metric = globals()[metric]
    except KeyError as e:
            metric = globals()[metric + '_distance']
    if isinstance(encoding, str):
        try:
            encoding = globals()[encoding]
        except KeyError:
            if encoding == 'identity':
                encoding = lambda s: s
            else:
                raise ValueError("unknown encoding function '%s'" % encoding)
    e_known_word = unicode(encoding(known_word))
    e_unknown_word = unicode(encoding(unknown_word))
    if verbose:
        print(metric, e_known_word, e_unknown_word)
    return metric(e_known_word, e_unknown_word)

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
