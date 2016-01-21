import warnings
warnings.filterwarnings("error")

import operator
import cPickle

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

def build_features_target(df):
    feature_names = df.columns.tolist()
    print('feature_names', feature_names)
    feature_names = feature_names[13:]
    feature_names.remove('suggestion')
    feature_names = feature_names[0:-1]
    target_name = 'target'
    print('feature_names', feature_names)
    return feature_names, target_name

def add_count_features(df, vectorizer, column, prefix):
    count_features = vectorizer.transform(df[column]).todense()
    rindex = [(v,k) for k,v in vectorizer.vocabulary_]
    sorted_rindex = sorted(rindex,
            key=operator.itemgetter(0))
    feature_names = [prefix+t[1] for t in sorted_rindex]
    for i,feature_name in enumerate(feature_names):
        df.loc[:, feature_name] = count_features[:, i]

def fit_cv(estimator, df_train, df_valid=None, train_size=None, dictionary='Aspell', scale=False, correct_word_is_in_suggestions=False, random_state=17, use_ngrams=True, verbose=True):

    feature_names, target_name = build_features_target(df_train)

    if df_valid is None:
        errors = df_train.error.unique().tolist()
        train_errors, valid_errors = train_test_split(errors,
                train_size=0.9, random_state=random_state)
        if verbose:
            print('train_errors %d valid_errors %d' %
                (len(train_errors), len(valid_errors)))
        df_valid = df_train[df_train.error.isin(valid_errors)].copy()
        df_train = df_train[df_train.error.isin(train_errors)].copy()

    def build_vocab_mask(df):
        if correct_word_is_in_suggestions:
            return df.correct_word_is_in_suggestions
        else:
            return df.correct_word_in_dict

    df_valid = df_valid.copy()

    if verbose:
        print('train %d valid %d' % (len(df_train), len(df_valid)))
    
    if use_ngrams:
        # Fit the error and suggestion vectorizers, add features to
        # data frames.
        err_vec = CountVectorizer(ngram_range=(2, 2), analyzer=u'char')
        sugg_vec = CountVectorizer(ngram_range=(2, 2), analyzer=u'char')

        err_vec.fit(df_train.error)
        add_count_features(df_train, err_vec, 'error', 'error_')
        add_count_features(df_valid, err_vec, 'error', 'error_')

        sugg_vec.fit(df_train.suggestion)
        add_count_features(df_train, sugg_vec, 'suggestion',
            'suggestion_')
        add_count_features(df_valid, sugg_vec, 'suggestion',
            'suggestion_')

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

    correct_top_1 = []

    # Some errors might not be unique for a given correct word
    # (i.e. ther -> their, ther -> there), so when getting the
    # validation set predictions, partition the examples by
    # correct word and error.
    correct_words = df_valid.correct_word.unique()
    pbar = build_progressbar(correct_words)
    y_hat_valid = np.zeros(len(df_valid))
    for i,correct_word in enumerate(df_valid.correct_word.unique()):
        cw_mask = df_valid.correct_word == correct_word
        errors = df_valid[cw_mask].error.unique()
        pbar.update(i+1)
        if verbose:
            print('  CORRECT WORD %s %d %d' % (
                    correct_word, cw_mask.sum(), len(errors)))
            print('  ERRORS ' + str(errors))
        for j,error in enumerate(errors):
            if verbose:
                print('  CORRECT WORD %s ERROR %s' % (correct_word, error))
            mask = (cw_mask & (df_valid.error == error)).values
            df_valid_tmp = df_valid[mask]
            if verbose:
                print('df_valid_tmp', df_valid_tmp.shape)
            y_valid_tmp = y_valid[mask].values
            y_valid_tmp_proba = estimator.predict_proba(X_valid[mask])
            if verbose:
                print('mask', type(mask), mask.shape)
                print('y_hat_valid', type(y_hat_valid), y_hat_valid.shape)
            y_hat_valid[mask] = estimator.predict(X_valid[mask])

            y_valid_tmp_pred = np.zeros_like(y_valid_tmp)

            top_1 = np.argmax(y_valid_tmp_proba[:, 1])
            y_valid_tmp_pred[top_1] = 1

            new_suggestion_index = np.ones_like(y_valid_tmp)
            new_suggestion_index[top_1] = 0
            df_valid.loc[mask, 'suggestion_index'] = new_suggestion_index

            if np.all(y_valid_tmp == y_valid_tmp_pred):
                correct_top_1.append(1)
            else:
                correct_top_1.append(0)
    pbar.finish()

    print('top 1 accuracy %d/%d %0.4f' % (
        sum(correct_top_1),
        float(len(correct_top_1)),
        sum(correct_top_1) / float(len(correct_top_1))))

    return df_valid, y_hat_valid

def run_cv_one_dataset(train_size=0.9, k=1, seed=17, n_jobs=5, suggestions_from='Aspell', verbose=False):
    dfs_paths = [
            'data/aspell.pkl', 'data/birbeck.pkl', 
            'data/holbrook-missp.pkl', 'data/wikipedia.pkl'
            ]

    rf = RandomForestClassifier(random_state=seed, n_jobs=n_jobs, class_weight='balanced')

    for dfs_path in dfs_paths:
        dfs = cPickle.load(open(dfs_path))
        remove_non_aspell_vocabulary_dicts(dfs)
        df = dfs[suggestions_from]
        print(dfs_path)
        fit_cv(rf, df, train_size=train_size, verbose=verbose)
        dict_df = spelling.mitton.evaluate_ranks(dfs, ranks=[1], verbose=True)
        print(dict_df.sort_values('Accuracy').tail(k))

def run_leave_out_one_dataset(n_jobs, k=1, seed=17, suggestions_from='Aspell', verbose=False):
    dfs_paths = [
            'data/aspell.pkl', 'data/birbeck.pkl', 
            'data/holbrook-missp.pkl', 'data/wikipedia.pkl'
            ]

    rf = RandomForestClassifier(random_state=seed, n_jobs=n_jobs, class_weight='balanced')

    dfs = {}

    for dfs_path in dfs_paths:
        dataset_name = dfs_path.replace('data/', '')
        dataset_name = dataset_name.replace('.csv', '')

        if verbose:
            print('loading ' + dfs_path)

        dfs_tmp = cPickle.load(open(dfs_path))
        remove_non_aspell_vocabulary_dicts(dfs_tmp)
        dfs[dataset_name] = dfs_tmp

    results = {}

    for held_out in dfs.keys():
        print(held_out)
        if verbose:
            print('dfs', len(dfs))
        tmp_dfs = dict(dfs)
        if verbose:
            print('tmp_dfs (before del)', type(tmp_dfs), len(tmp_dfs))
        del tmp_dfs[held_out]
        if verbose:
            print('tmp_dfs (after del)', type(tmp_dfs), len(tmp_dfs))

        held_out_dfs = dfs[held_out]
        if verbose:
            print('held_out_dfs', held_out_dfs.keys())
        df_valid = held_out_dfs[suggestions_from]
        
        train_dfs = [dfs[dataset][suggestions_from] for dataset in tmp_dfs.keys()]
        df_train = pd.concat(train_dfs)

        if verbose:
            print(df_train.shape, df_valid.shape)

        df_valid_reranked, y_hat = fit_cv(rf, df_train, df_valid, verbose=verbose)
        held_out_dfs['RandomForest'] = df_valid_reranked
        results[held_out] = dict(held_out_dfs)
        dict_df = spelling.mitton.evaluate_ranks(held_out_dfs, ranks=[1], verbose=True)
        print(dict_df.sort_values('Accuracy'))
    return results
