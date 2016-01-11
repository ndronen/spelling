from __future__ import absolute_import

import collections
import enchant
import progressbar
import pandas as pd
from spelling.features import (suggest,
        compute_unary_features, compute_binary_features)
from spelling.dictionary import (Aspell, Norvig, 
        AspellWithNorvigLanguageModel, NorvigWithoutLanguageModel,
        NorvigWithAspellDictWithoutLanguageModel,
        NorvigWithAspellDictAndGoogleLanguageModel)

from spelling.dictionary import NORVIG_DATA_PATH

from .utils import build_progressbar

def load_mitton_words(path):
    with open(path) as f:
        mitton_words = [w.strip() for w in f]
    return mitton_words

def build_mitton_pairs(words):
    correct_word = None
    pairs = []
    
    for word in words:
        if word.startswith('$'):
            correct_word = word[1:]
            continue

        # In holbrook-missp.dat, there are 20 misspellings with unknown
        # corrections listed under the correct word '?'.  Skip them.
        if correct_word == '?':
            continue

        # In holbrook-missp.dat, a line containing a misspelling
        # has MISSPELLING SPACE COUNT.  Remove the COUNT.
        word = word.split(' ')[0]

        pairs.append((word, correct_word))

    return pairs

def build_dictionary(constructor, lang='en_US'):
    return constructor(lang)

def build_progressbar(items):
    return progressbar.ProgressBar(term_width=40,
        widgets=[' ', progressbar.Percentage(),
        ' ', progressbar.ETA()],
        maxval=len(items)).start()

def build_dataset(pairs, dictionary):
    dataset = []
    pbar = build_progressbar(pairs)
    row = {}
    for i, (error, correct_word) in enumerate(pairs):
        pbar.update(i+1)

        row['error'] = error
        row['correct_word'] = correct_word
        row['error_is_real_word'] = dictionary.check(error)

        # Add unary and binary features for error and the correct word.
        for k,v in compute_unary_features(error).iteritems():
            k = 'error_' + k
            row[k] = v

        for k,v in compute_binary_features(error, correct_word).iteritems():
            k = 'correct_word_' + k
            row[k] = v

        # Get the dictionary suggestions and compute unary and binary features with each suggestion.
        suggestions = suggest(dictionary, error)
        row['suggestion_count'] = len(suggestions)

        row['correct_word_in_dict'] = dictionary.check(correct_word)
        row['correct_word_is_in_suggestions'] = correct_word in suggestions

        try:
            row['correct_words_suggestions_index'] = suggestions.index(correct_word)
        except ValueError:
            row['correct_words_suggestions_index'] = -1

        for i, suggestion in enumerate(suggestions):
            for k,v in compute_unary_features(suggestion).iteritems():
                k = 'suggestion_' + k
                row[k] = v

            for k,v in compute_binary_features(error, suggestion).iteritems():
                k = 'suggestion_' + k
                row[k] = v

            row['suggestion'] = suggestion
            row['suggestion_index'] = i
            row['target'] = 1 if suggestion == correct_word else 0
            row['same_first_char_error_suggestion'] = error[0] == suggestion[0]
            row['suggestion_lm_log_prob'] = 0.

            # Add defensively-copied row to dataset.
            dataset.append(dict(row))

    pbar.finish()

    df = pd.DataFrame(dataset)
    return df[df.columns.sort_values()]

    """
    cols = [
            'error',
            'error_is_real_word',
            'correct_word',
            'suggestion',
            'suggestion_index',
            'target',

            'correct_word_in_dict',
            'correct_word_is_in_suggestions',
            'correct_word_suggestions_index',

            'correct_word_levenshtein_distance',
            'correct_word_keyboard_distance',
            'correct_word_soundex_equal',
            'correct_word_metaphone_equal',

            'suggestion_levenshtein_distance',
            'suggestion_keyboard_distance',
            'suggestion_soundex_equal',
            'suggestion_metaphone_equal',

            'error_char_count',
            'error_consonant_count',
            'error_vowel_count',
            'error_capital_count',

            'suggestion_char_count',
            'suggestion_consonant_count',
            'suggestion_vowel_count',
            'suggestion_capital_count',

            'suggestion_contains_space',
            'same_first_char_error_suggestion',

            'suggestion_lm_log_prob'
            ]
    return df[cols]
    """
 
CONSTRUCTORS = [
        Aspell, Norvig, AspellWithNorvigLanguageModel,
        NorvigWithoutLanguageModel,
        NorvigWithAspellDictWithoutLanguageModel,
        NorvigWithAspellDictAndGoogleLanguageModel
        ]

def build_datasets(pairs, constructors=CONSTRUCTORS):
    """
    From a list consisting of pairs of known words and misspellings,
    build data frames of features derived from dictionary suggestions
    for the misspellings.

    Parameters
    ----------
    pairs : iterable of tuple
      An iterable of tuples of (known word, misspelled word) pairs.
    constructors : iterable of classes
      An iterable of class names for constructing dictionaries.

    Returns
    ----------
    datasets : dict of pandas.DataFrame
      A dictionary of data frames, with one key for each dictionary.
    """
    datasets = {}
    for constructor in constructors:
        dictionary = build_dictionary(constructor)
        dataset = build_dataset(pairs, dictionary)
        datasets[constructor.__name__] = dataset
    return datasets

def build_mitton_datasets(path, constructors=CONSTRUCTORS):
    """
    Build data frames of features derived from dictionary suggestions
    for one of the Roger Mitton spelling error datasets in our
    data directory.

    Parameters
    ----------
    path : str
      Path to a spelling error dataset in Roger Mitton's format.
    constructors : iterable of classes
      An iterable of class names for constructing dictionaries.

    Returns
    ----------
    datasets : dict of pandas.DataFrame
      A dictionary of data frames, with one key for each dictionary.
    """
    mitton_words = load_mitton_words(path)
    pairs = build_mitton_pairs(mitton_words)
    return build_datasets(pairs, constructors)

def evaluate_ranks(dfs, ranks=[1, 2, 3, 4, 5, 10, 25, 50], verbose=False):
    """
    Evaluate the accuracy of one or more dictionaries using data frames
    created by `build_datasets` or `build_mitton_datasets`.

    Parameters
    ----------
    dfs : dict of pandas.DataFrame
      A dictionary of data frames, with one key for each dictionary.
    ranks : list of int
      The ranks at which to evaluate accuracy.  Top-5 rank, for example,
      measures accuracy as the correct word being one of a dictionary's 
      first five suggested corrections.
    verbose : bool
      Whether to print some extra information during execution.

    Returns
    -------
    df : a pandas.DataFrame
      The accuracy of each dictionary at a given rank.
    """
    dict_names = sorted(dfs.keys())

    all_words = set()
    common_words = set()

    df = dfs[dict_names[0]]
    common_words.update(
            df.correct_word[df.correct_word_in_dict])
    all_words.update(common_words)

    for dict_name in dict_names[1:]:
        df = dfs[dict_name]
        common_words.intersection_update(
            df.correct_word[df.correct_word_in_dict])
        all_words.update(
            df.correct_word[df.correct_word_in_dict])

    if verbose:
        print('words %d' % len(all_words))
        print('common words %d' % len(common_words))

    accuracies = collections.defaultdict(list)

    for dict_name in dict_names:
        df = dfs[dict_name]
        df = df[df.correct_word.isin(common_words)]
        n = float(len(df[df.suggestion_index == 0]))
        for rank in ranks:
            n_correct = len(df[(df.suggestion_index < rank) & (df.suggestion == df.correct_word)])
            accuracies[dict_name].append(n_correct/n)

    evalutation = collections.defaultdict(list)

    for dict_name in sorted(dfs.keys()):
        for i, rank in enumerate(ranks):
            evalutation['Algorithm'].append(dict_name)
            evalutation['Accuracy'].append(accuracies[dict_name][i])
            evalutation['Rank'].append(rank)

    return pd.DataFrame(evalutation)[['Algorithm', 'Rank', 'Accuracy']]

