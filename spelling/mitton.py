from __future__ import absolute_import

import codecs
import collections
import enchant
import progressbar
import pandas as pd
from spelling.features import (suggest,
        compute_unary_features, compute_binary_features)
from spelling.dictionary import (
        build_norvig, build_aspell,
        build_norvig_without_language_model,
        build_norvig_with_aspell_vocab_without_language_model,
        build_norvig_with_aspell_vocab_with_language_model_sorter,
        build_aspell_with_language_model_sorter,
        build_aspell_vocab_with_metaphone_retriever_and_language_model_sorter,
        build_aspell_vocab_with_nn_retriever_and_language_model_sorter)

from spelling.dictionary import NORVIG_DATA_PATH

PROBS_DATA_PATH = 'data/aspell-dict.csv.gz'

from .utils import build_progressbar

CORPORA = [
        'data/aspell.dat', 'data/birbeck.dat',
        'data/holbrook-missp.dat',
        'data/wikipedia.dat'
        ]

def load_mitton_words(path):
    with codecs.open(path, 'r', encoding='utf8') as f:
        mitton_words = [w.strip() for w in f]
    return mitton_words

def build_mitton_pairs(words, error_first=True):
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

        if '_' in correct_word:
            correct_word = correct_word.replace('_', ' ')

        if error_first:
            pair = (word, correct_word)
        else:
            pair = (correct_word, word)

        pairs.append(pair)

    return pairs

def build_progressbar(items):
    return progressbar.ProgressBar(term_width=40,
        widgets=[' ', progressbar.Percentage(),
        ' ', progressbar.ETA()],
        maxval=len(items)).start()

def build_probs_dict(probs_data_path=PROBS_DATA_PATH):
    df = pd.read_csv(probs_data_path, sep='\t', encoding='utf8')
    probs = collections.defaultdict(float)
    probs.update(dict(zip(df.word, df.google_ngram_prob)))
    return probs

def build_dataset(pairs, dictionary, probs=build_probs_dict(), verbose=False):
    dataset = []
    pbar = build_progressbar(pairs)
    row = {}
    for i, (error, correct_word) in enumerate(pairs):
        pbar.update(i+1)

        try:
            iter(error)
        except TypeError:
            if verbose:
                print('skipping non-iterable misspelling ' + str(error))
            continue

        try:
            error.encode('ascii')
        except (UnicodeDecodeError, UnicodeEncodeError):
            if verbose:
                print('skipping non-ASCII misspelling ' + str(error))
            continue

        try:
            iter(correct_word)
        except TypeError:
            if verbose:
                print('skipping non-iterable correction ' + str(correct_word))
            continue

        try:
            correct_word.encode('ascii')
        except (UnicodeDecodeError, UnicodeEncodeError):
            if verbose:
                print('skipping non-ASCII correction ' + str(correct_word))
            continue

        row['error'] = error
        row['error_prob'] = probs[error]
        row['correct_word'] = correct_word
        row['correct_word_prob'] = probs[correct_word]
        row['error_is_real_word'] = dictionary.check(error)

        # Add unary and binary features for error and the correct word.
        for k,v in compute_unary_features(error).iteritems():
            k = 'error_' + k
            row[k] = v

        for k,v in compute_binary_features(correct_word, error).iteritems():
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

        i = -1
        for suggestion in suggestions:
            ###############################################################
            # We need to compute keyboard distance, so require suggestions
            # to contain only characters that exist in the QWERTY keyboard.
            ###############################################################
            try:
                iter(suggestion)
            except TypeError:
                if verbose:
                    print('skipping non-iterable suggestion ' + str(suggestion))
                continue
    
            try:
                suggestion.encode('ascii')
            except (UnicodeDecodeError, UnicodeEncodeError):
                if verbose:
                    print('skipping non-ASCII suggestion ' + str(suggestion))
                continue

            i += 1

            for k,v in compute_unary_features(suggestion).iteritems():
                k = 'suggestion_' + k
                row[k] = v

            for k,v in compute_binary_features(suggestion, error).iteritems():
                k = 'suggestion_' + k
                row[k] = v

            row['suggestion'] = suggestion
            row['suggestion_index'] = i
            row['target'] = 1 if suggestion == correct_word else 0
            row['same_first_char_error_suggestion'] = error[0] == suggestion[0]
            row['suggestion_prob'] = probs[suggestion]

            # Add defensively-copied row to dataset.
            dataset.append(dict(row))

    pbar.finish()

    df = pd.DataFrame(dataset)
    return df[df.columns.sort_values()]

CONSTRUCTORS = [
        build_norvig,
        build_aspell,
        build_norvig_without_language_model,
        build_norvig_with_aspell_vocab_without_language_model,
        build_norvig_with_aspell_vocab_with_language_model_sorter,
        build_aspell_with_language_model_sorter,
        build_aspell_vocab_with_metaphone_retriever_and_language_model_sorter,
        build_aspell_vocab_with_nn_retriever_and_language_model_sorter
        ]

def build_datasets(pairs, constructors=CONSTRUCTORS, verbose=False):
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

        dataset = build_dataset(pairs, dictionary, verbose=verbose)
        datasets[constructor.__name__] = dataset
    return datasets

def build_mitton_datasets(path, constructors=CONSTRUCTORS, verbose=False):
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
    return build_datasets(pairs, constructors, verbose=verbose)

def evaluate_ranks(dfs, ranks=[1, 2, 3, 4, 5, 10, 25, 50], ignore_case=False, correct_word_is_in_suggestions=False, verbose=True):
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

    # Defensively copy data frames before modifying them.
    if ignore_case:
        df_copies = {}
        for dict_name in dict_names:
            df = dfs[dict_name]
            df = df.copy()
            df_copies[dict_name] = df

            df.suggestion = df.suggestion.apply(lambda s: s.lower())
            df.correct_word = df.correct_word.apply(lambda s: s.lower())
        dfs = df_copies

    all_words = set()
    common_words = set()


    def build_vocab_mask(df):
        if correct_word_is_in_suggestions:
            return df.correct_word_is_in_suggestions
        else:
            return df.correct_word_in_dict

    # Find the set of common words with which to evaluate the performance
    # of the dictionaries.  Start with the words in the first data frame
    # for which the correct word is either in the suggestions or in
    # the dictionary, depending on the value of the parameter
    # correct_word_is_in_suggestions.  From there, repeatedly take the
    # intersection of the set of common words so far and the words 
    # in the next data frame that are either in the suggestions or
    # in the dictionary.
    df = dfs[dict_names[0]]
    common_words.update(
            df.correct_word[build_vocab_mask(df)])
    all_words.update(common_words)

    for dict_name in dict_names[1:]:
        df = dfs[dict_name]
        common_words.intersection_update(
            df.correct_word[build_vocab_mask(df)])
        all_words.update(
            df.correct_word[build_vocab_mask(df)])

    if verbose:
        print('words %d' % len(all_words))
        print('common words %d' % len(common_words))

    accuracies = collections.defaultdict(list)

    for dict_name in dict_names:
        df = dfs[dict_name]
        df = df[df.correct_word.isin(common_words)]
        n = float(len(df[df.suggestion_index == 0]))
        for rank in ranks:
            # If the dictionary's suggestion is the correct word and
            # the position of the suggestion in the dictionary's
            # suggestion list is less than the given rank, then
            # the dictionary is said to have gotten the correction
            # right at that rank.  The number correct at this rank
            # is cumulative.
            n_correct = len(df[(df.suggestion_index < rank) & (df.suggestion == df.correct_word)])
            accuracies[dict_name].append(n_correct/n)

    evaluation = collections.defaultdict(list)

    for dict_name in sorted(dfs.keys()):
        for i, rank in enumerate(ranks):
            evaluation['Algorithm'].append(dict_name)
            evaluation['Accuracy'].append(accuracies[dict_name][i])
            evaluation['Rank'].append(rank)

    return pd.DataFrame(evaluation)[['Algorithm', 'Rank', 'Accuracy']]
