import warnings
warnings.filterwarnings("error")

import os
import operator
import pickle

import numpy as np
import pandas as pd

import sklearn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.feature_extraction.text import CountVectorizer

import spelling.mitton
from spelling.utils import build_progressbar

def load_data(path, dictionary='Aspell'):
    dfs = cPickle.load(open(path))
    return dfs[dictionary]

def remove_non_aspell_vocabulary_dicts(dfs):
    non_aspell_vocab_dicts = ['Norvig', 'NorvigWithoutLanguageModel']
    for d in non_aspell_vocab_dicts:
        try:
            del dfs[d]
        except KeyError:
            pass

def build_features_target(df, remove_suggestion_index=True):
    feature_names = []
    for feature_name in df.columns:
        if feature_name.startswith('correct_word'):
            continue
        feature_names.append(feature_name)
    feature_names.remove('error')
    feature_names.remove('suggestion')
    feature_names.remove('target')
    if remove_suggestion_index:
        feature_names.remove('suggestion_index')
    target_name = 'target'
    return feature_names, target_name

def add_features_from_vectorizer(df, vectorizer, column, feature_name_prefix=None):
    if feature_name_prefix is None:
        feature_name_prefix = column + '_'
    count_features = np.array(vectorizer.transform(df[column]).todense())
    rindex = [(v,k) for k,v in vectorizer.vocabulary_.iteritems()]
    sorted_rindex = sorted(rindex,
            key=operator.itemgetter(0))
    feature_names = [feature_name_prefix+t[1] for t in sorted_rindex]

    pbar = build_progressbar(feature_names)
    for i,feature_name in enumerate(feature_names):
        pbar.update(i+1)
        df.loc[:, feature_name] = count_features[:, i]
    pbar.finish()
    return feature_names

def add_ngram_features(df_train, df_valid, column):
    vectorizer = CountVectorizer(
            ngram_range=(2, 2), analyzer=u'char',
            min_df=5, max_features=200)
    vectorizer.fit(df_train[column])
    ngram_feature_names = add_features_from_vectorizer(
            df_train, vectorizer, column)
    add_features_from_vectorizer(df_valid, vectorizer, column)
    return ngram_feature_names

def split_train(df_train, train_size, random_state=17):
    errors = df_train.error.unique().tolist()
    train_errors, valid_errors = train_test_split(errors,
            train_size=0.9, random_state=random_state)
    if verbose:
        print('train_errors %d valid_errors %d' %
            (len(train_errors), len(valid_errors)))
    df_valid = df_train[df_train.error.isin(valid_errors)].copy()
    df_train = df_train[df_train.error.isin(train_errors)].copy()
    return df_train, df_valid

def prepare_data(df_train, df_valid=None, train_size=0.9, use_ngrams=True, remove_suggestion_index=True, random_state=17, verbose=True):
    feature_names, target_name = build_features_target(df_train,
            remove_suggestion_index=remove_suggestion_index)

    if df_valid is None:
        if verbose:
            print('splitting df_train into train and valid')
        df_train, df_valid = split_train(df_train,
                train_size=train_size, random_state=random_state)

    df_valid = df_valid.copy()

    if verbose:
        print('train %d valid %d' % (len(df_train), len(df_valid)))

    if use_ngrams:
        # Fit the error and suggestion vectorizers, add features to
        # data frames.
        if verbose:
            print('adding "error_" features')
            feature_names.extend(add_ngram_features(
                    df_train, df_valid, 'error'))
        if verbose:
            print('adding "suggestion_" features')
            feature_names.extend(add_ngram_features(
                    df_train, df_valid, 'suggestion'))

    return df_train, df_valid, feature_names, target_name

def fit_cv(estimator, df_train, df_valid, feature_names, target_name, scale=False, correct_word_is_in_suggestions=False, random_state=17, verbose=True):

    df_valid = df_valid.copy()

    X_train = df_train[feature_names]
    X_valid = df_valid[feature_names]
    
    y_train = df_train[target_name]
    y_valid = df_valid[target_name]

    if verbose:
        print('train %d valid %d' % (len(X_train), len(X_valid)))

    scaler = StandardScaler()
    if scale:
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)

    estimator.fit(X_train, y_train)
    if verbose:
        print(estimator)

    # Some errors might not be unique for a given correct word
    # (i.e. ther -> their, ther -> there), so when getting the
    # validation set predictions, partition the examples by
    # correct word and error.
    correct_words = df_valid.correct_word.unique()
    pbar = build_progressbar(correct_words)
    y_hat_valid_proba = estimator.predict_proba(X_valid)
    for i,correct_word in enumerate(df_valid.correct_word.unique()):
        cw_mask = df_valid.correct_word == correct_word
        errors = df_valid[cw_mask].error.unique()
        pbar.update(i+1)
        for j,error in enumerate(errors):
            mask = (cw_mask & (df_valid.error == error)).values
            y_valid_tmp = y_valid[mask].values
            y_valid_tmp_proba = y_hat_valid_proba[mask]

            start = len(y_valid_tmp) - 1
            stop = -1
            step = -1
            ranks = np.argsort(y_valid_tmp_proba[:, 1])
            indices = np.arange(start, stop, step)
            y_valid_tmp_pred = np.zeros_like(y_valid_tmp)
            y_valid_tmp_pred[ranks] = np.arange(start, stop, step)

            new_suggestion_index = np.ones_like(y_valid_tmp)
            new_suggestion_index = y_valid_tmp_pred
            df_valid.loc[mask, 'suggestion_index'] = new_suggestion_index

    pbar.finish()

    return df_valid

def run_cv_one_dataset(train_size=0.9, k=1, seed=17, n_jobs=5, suggestions_from='Aspell', verbose=False, remove_suggestion_index=True, dfs_paths=['data/aspell.pkl', 'data/birbeck.pkl', 'data/holbrook-missp.pkl', 'data/wikipedia.pkl']):

    rf = RandomForestClassifier(n_jobs=n_jobs, random_state=seed,
            class_weight='balanced')

    for dfs_path in dfs_paths:
        dfs = cPickle.load(open(dfs_path))
        remove_non_aspell_vocabulary_dicts(dfs)
        df = dfs[suggestions_from]
        print(dfs_path)
        fit_cv(rf, df, train_size=train_size, verbose=verbose,
                remove_suggestion_index=remove_suggestion_index)
        dict_df = spelling.mitton.evaluate_ranks(dfs, ranks=[1], verbose=True)
        print(dict_df.sort_values('Accuracy').tail(k))

def run_dataset(estimator, dataset, random_state=17, verbose=False, **kwargs):
    # fit_cv kwargs:
    # scale=False
    # correct_word_is_in_suggestions=False
    # random_state=17
    # verbose=True
    return fit_cv(estimator, dataset['train'], dataset['valid'],
            dataset['feature_names'], dataset['target_name'], **kwargs)

def run_datasets(estimator, datasets, random_state=17, verbose=False, estimator_name=None, **kwargs):
    if estimator_name is None:
        estimator_name = estimator.__class__.__name__
    for ds_name in datasets.keys():
        print(ds_name)
        datasets[ds_name]['dicts'][estimator_name] = run_dataset(estimator, datasets[ds_name])

def prepare_leave_out_one_datasets(suggestions_from='Aspell', dfs_paths=['data/aspell.pkl', 'data/birbeck.pkl', 'data/holbrook-missp.pkl', 'data/wikipedia.pkl'], max_train_size=None, verbose=True):

    dfs = {}

    for dfs_path in dfs_paths:
        dataset_name = dfs_path.replace('data/', '')
        dataset_name = dataset_name.replace('.csv', '')

        if verbose:
            print('loading ' + dfs_path)

        dfs_tmp = cPickle.load(open(dfs_path))
        remove_non_aspell_vocabulary_dicts(dfs_tmp)
        dfs[dataset_name] = dfs_tmp

    datasets = {}

    for held_out in dfs.keys():
        if verbose:
            print(held_out)
        tmp_dfs = dict(dfs)
        del tmp_dfs[held_out]
        held_out_dfs = dfs[held_out]
        df_valid = held_out_dfs[suggestions_from]
        
        train_dfs = [dfs[dataset][suggestions_from] for dataset in tmp_dfs.keys()]
        df_train = pd.concat(train_dfs)

        if max_train_size is not None:
            assert max_train_size > 1
            df_train = df_train.ix[0:max_train_size, :]

        if verbose:
            print('before pruning training %s %s ' %
                    (str(df_train.shape), str(df_valid.shape)))

        valid_errors = df_valid.error
        df_train = df_train[~df_train.error.isin(valid_errors)]

        if verbose:
            print('after pruning training %s %s ' %
                    (str(df_train.shape), str(df_valid.shape)))
            print('calling prepare_data')

        df_train, df_valid, feature_names, target = prepare_data(
                df_train, df_valid, train_size=0.9, verbose=verbose)

        if verbose:
            print('done with prepare_data')

        held_out_name = os.path.splitext(held_out)[0]
        print(held_out_name)
        datasets[held_out_name] = {
                    'train': df_train,
                    'valid': df_valid,
                    'dicts': held_out_dfs,
                    'feature_names': feature_names,
                    'target_name': target
                }

    return datasets
