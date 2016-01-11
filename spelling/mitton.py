from __future__ import absolute_import

import collections
import enchant
import progressbar
import pandas as pd
import spelling.features 
from spelling.dictionary import (Aspell, Norvig, 
        AspellWithNorvigLanguageModel, NorvigWithoutNorvigLanguageModel,
        NorvigWithAspellDictGoogleLanguageModel)

from spelling.dictionary import NORVIG_DATA_PATH

from .utils import build_progressbar

def load_mitton_words(path):
    with open(path) as f:
        mitton_words = [w.strip() for w in f]
    return mitton_words

def build_errors_correction_pairs(words):
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
    for i, (error, correct_word) in enumerate(pairs):
        pbar.update(i+1)

        real_word_error = dictionary.check(error)

        dsugs = spelling.features.suggestions(dictionary, error)
        n_dsugs = len(dsugs)

        correct_word_in_dict = dictionary.check(correct_word)
        correct_word_in_dsugs = correct_word in dsugs
        try:
            correct_words_dsugs_index = dsugs.index(correct_word)
        except ValueError:
            correct_words_dsugs_index = -1

        corwd_ldist = spelling.features.levenshtein_distance(error, correct_word)
        corwd_kdist = spelling.features.keyboard_distance(error, correct_word)
        corwd_se = spelling.features.soundex_equal(error, correct_word)
        corwd_me = spelling.features.metaphone_equal(error, correct_word)

        for i, dsug in enumerate(dsugs):
            dataset.append( {
                'error': error,
                'real_word_error': real_word_error,
                'correct_word': correct_word,
                'dsug': dsug,
                'dsug_index': i,
                'target': 1 if dsug == correct_word else 0,

                'correct_word_in_dict': correct_word_in_dict,
                'correct_word_in_dsugs': correct_word_in_dsugs,
                'correct_word_dsugs_index': correct_words_dsugs_index,
                'n_dsugs': n_dsugs,

                'corwd_ldist': corwd_ldist,
                'corwd_kdist': corwd_kdist,
                'corwd_se': corwd_se,
                'corwd_me': corwd_me,

                'dsug_ldist': spelling.features.levenshtein_distance(error, dsug),
                'dsug_kdist': spelling.features.keyboard_distance(error, dsug),
                'dsug_se': spelling.features.soundex_equal(error, dsug),
                'dsug_me': spelling.features.metaphone_equal(error, dsug),

                'error_char_count': spelling.features.character_count(error),
                'error_consonant_count': spelling.features.consonant_count(error),
                'error_vowel_count': spelling.features.vowel_count(error),
                'error_capital_count': spelling.features.capital_count(error),

                'dsug_char_count': spelling.features.character_count(dsug),
                'dsug_consonant_count': spelling.features.consonant_count(dsug),
                'dsug_vowel_count': spelling.features.vowel_count(dsug),
                'dsug_capital_count': spelling.features.capital_count(dsug),

                'dsug_contains_space': spelling.features.contains_space(dsug),
                'same_first_char_error_dsug': error[0] == dsug[0],

                'dsug_lm_log_prob': 0.
                })

    pbar.finish()

    df = pd.DataFrame(dataset)

    cols = [
            'error',
            'real_word_error',
            'correct_word',
            'dsug',
            'dsug_index',
            'target',

            'correct_word_in_dict',
            'correct_word_in_dsugs',
            'correct_word_dsugs_index',

            'corwd_ldist',
            'corwd_kdist',
            'corwd_se',
            'corwd_me',

            'dsug_ldist',
            'dsug_kdist',
            'dsug_se',
            'dsug_me',

            'error_char_count',
            'error_consonant_count',
            'error_vowel_count',
            'error_capital_count',

            'dsug_char_count',
            'dsug_consonant_count',
            'dsug_vowel_count',
            'dsug_capital_count',

            'dsug_contains_space',
            'same_first_char_error_dsug',

            'dsug_lm_log_prob'
            ]

    return df[cols]
 
CONSTRUCTORS = [
        Aspell, Norvig, AspellWithNorvigLanguageModel,
        NorvigWithoutNorvigLanguageModel,
        NorvigWithAspellDictGoogleLanguageModel
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
    pairs = build_errors_correction_pairs(mitton_words)
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
        n = float(len(df[df.dsug_index == 0]))
        for rank in ranks:
            n_correct = len(df[(df.dsug_index < rank) & (df.dsug == df.correct_word)])
            accuracies[dict_name].append(n_correct/n)

    evalutation = collections.defaultdict(list)

    for dict_name in sorted(dfs.keys()):
        for i, rank in enumerate(ranks):
            evalutation['Algorithm'].append(dict_name)
            evalutation['Accuracy'].append(accuracies[dict_name][i])
            evalutation['Rank'].append(rank)

    return pd.DataFrame(evalutation)[['Algorithm', 'Rank', 'Accuracy']]

