import os
import collections
import h5py
import itertools
import modeling.utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (confusion_matrix, classification_report)

BASE_DIR = os.environ['HOME'] + '/proj/spelling/'
MODEL_DIR = BASE_DIR + '/models/keras/spelling/convnet/exp03-inputs/'
#DATA_DIR = BASE_DIR + '/data/spelling/experimental/'
DATA_DIR = BASE_DIR + '/results/exp03/'

MODEL_TEMPLATE = 'op_%s_n_ops_%d_n_errors_per_word_%d'
DATA_TEMPLATE = 'op-%s-distance-%d-errors-per-word-%d'
PRED_TEMPLATE = DATA_TEMPLATE + '-model-%s-%d-%d'

OPS = ['transpose', 'substitute', 'insert', 'delete']

def build_csv_path(op, n_ops, n_errors_per_example):
    return DATA_DIR + (DATA_TEMPLATE % (op, n_ops, n_errors_per_example)) + '.csv'

def build_pred_path(op, n_ops, n_errors_per_example):
    return DATA_DIR + (PRED_TEMPLATE % (op, n_ops, n_errors_per_example, op, n_ops, n_errors_per_example)) + '-pred.h5'

def build_model_cfg_path(model_op, model_n_ops, model_n_errors_per_example, data_op, data_n_ops, data_n_errors_per_example):
    model_cfg_template = 'data/spelling/experimental/op-%s-distance-%d-errors-per-word-%d-model-%s-%d-%d-cfg.json'
    return model_cfg_template % (data_op, data_n_ops, data_n_errors_per_example,
            model_op, model_n_ops, model_n_errors_per_example)

def load_models_save_predictions(force=False):
    os.chdir(BASE_DIR)

    model_data_pairs = [
            #[['delete', 1, 3], ['delete', 1, 3]],
            #[['delete', 2, 3], ['delete', 2, 3]],
            #[['insert', 1, 3], ['insert', 1, 3]],
            #[['insert', 2, 3], ['insert', 2, 3]],
            [['substitute', 1, 3], ['substitute', 1, 3]],
            [['substitute', 2, 3], ['substitute', 2, 3]]
            ]
    
    for md_pair in model_data_pairs:
        model_op, model_n_ops, model_n_errors = md_pair[0]
        data_op, data_n_ops, data_n_errors = md_pair[1]

        model_path = MODEL_DIR + (MODEL_TEMPLATE % (model_op, model_n_ops, model_n_errors))
        data_path = DATA_DIR + (DATA_TEMPLATE % (data_op, data_n_ops, data_n_errors)) + '.h5'
        print(model_path, data_path)

        model_cfg_path = build_model_cfg_path(model_op, model_n_ops, model_n_errors,
                data_op, data_n_ops, data_n_errors)
        print("Checking for existence of %s" % model_cfg_path)
        if os.path.exists(model_cfg_path):
            print("%s exists" % model_cfg_path)
            if not force:
                print("Skipping job with model %s data %s because %s exists" % \
                        (model_path, data_path, model_cfg_path))
                continue
        else:
            print("%s doesn't exist" % model_cfg_path)

        model_name = 'model-' + '-'.join([str(x) for x in md_pair[0]])
        try:
            modeling.utils.load_predict_save(model_path, data_path,
                    model_name=model_name,
                    output_dir='/tmp/exp03/',
                    model_weights=True)
        except Exception as e:
            print(e)


def load_data(op, n_ops, n_errors_per_example):
    csv_path = build_csv_path(op, n_ops, n_errors_per_example)
    pred_path = build_pred_path(op, n_ops, n_errors_per_example)

    assert os.path.exists(csv_path)
    assert os.path.exists(pred_path)

    df = pd.read_csv(csv_path, sep='\t', encoding='utf8')
    pred_file = h5py.File(pred_path)

    prob = pred_file['prob'].value
    pred = pred_file['pred'].value

    df['pred'] = pred
    df['prob0'] = prob[:, 0]
    df['prob1'] = prob[:, 1]

    return df

def plot_histogram(op, n_ops, n_errors_per_example, ylim):
    df = load_data(op, n_ops, n_errors_per_example)
    plt.figure()
    plt.hist(df[df.binary_target == 1].prob1.values, alpha=0.5, color='blue', bins=100, range=(0,1))
    plt.hist(df[df.binary_target == 0].prob1.values, alpha=0.5, color='red', bins=100, range=(0,1))
    plt.ylim(0, ylim)
    plt.title("%s %d operations %d errors per example" % (op.title(), n_ops, n_errors_per_example))
    plt.show(block=False)

def build_confusion_matrix(df):
    return confusion_matrix(df.binary_target, df.pred)

def build_classification_report(df=None, digits=4):
    return classification_report(df.binary_target, df.pred, digits=digits)

def build_example_confusion_matrix(df):
    def subset_df(df, target, right, ascending, n=10):
        prob_name = 'prob%d' % target
        mask = df.pred == target
        if not right:
            mask = ~mask
        df = df[mask][df.binary_target == target]
        return df.sort_values(prob_name, ascending=ascending).head(n)

    # True negative
    # - binary_target = 0
    # - pred == binary_target
    # - show prob0
    # - sort prob0 ascending
    # - expected range of prob0 is 1.0-0.5
    df_tn = pd.concat([
            subset_df(df, 0, right=True, ascending=False),
            subset_df(df, 0, right=True, ascending=True)
        ], axis=0)
    df_tn = df_tn.sort_values('prob0', ascending=False)

    # False negative
    # - binary_target = 1
    # - pred != binary_target
    # - show prob0
    # - sort prob0 ascending
    # - expected range of prob0 is 0.5-1.0
    df_fn = pd.concat([
            subset_df(df, 1, right=False, ascending=False),
            subset_df(df, 1, right=False, ascending=True)
        ], axis=0)
    df_fn = df_fn.sort_values('prob0', ascending=True)

    # False positive
    # - binary_target = 0
    # - pred != binary_target
    # - show prob1
    # - sort prob1 ascending
    # - expected range of prob1 is 1.0-0.5
    df_fp = pd.concat([
            subset_df(df, 0, right=False, ascending=False),
            subset_df(df, 0, right=False, ascending=True)
        ], axis=0)
    df_fp = df_fp.sort_values('prob1', ascending=False)

    # True positive
    # - binary_target = 1
    # - pred == binary_target
    # - show prob1
    # - sort prob1 ascending
    # - expected range of prob1 is 0.5-1.0
    df_tp = pd.concat([
            subset_df(df, 1, right=True, ascending=False),
            subset_df(df, 1, right=True, ascending=True)
        ], axis=0)
    df_tp = df_tp.sort_values('prob1', ascending=True)

    # Concatenate true negative and false negative rows.
    df_neg = pd.concat([df_tn, df_fn], axis=0)
    df_neg = df_neg[['word', 'real_word', 'prob0']]
    df_neg.columns = [cn+'_neg' for cn in df_neg.columns]

    # Concatenate false positive and true positive rows.
    df_pos = pd.concat([df_fp, df_tp], axis=0)
    df_pos = df_pos[['word', 'real_word', 'prob1']]
    df_pos.columns = [cn+'_pos' for cn in df_pos.columns]

    #print(df_neg.head(1))
    #print(df_pos.head(1))

    # Concatenate the negative and positive data frames column-wise.
    for col in df_pos.columns:
        df_neg[col] = df_pos[[col]].values
    df_neg.columns = ['word', 'real_word', 'prob0', 'word', 'real_word', 'prob1']
    return df_neg

def build_confusion_matrices(n_ops, n_errors_per_example, ops=OPS, subsetter=lambda df: df):
    dfs = []
    for op in ops:
        df = load_data(op, n_ops, n_errors_per_example)
        df = subsetter(df)

        cm_df = pd.DataFrame(data=build_confusion_matrix(df))
        cm_df.index = [['Non-word', 'Real word']]
        cm_df.columns = ['Non-word', 'Real word']

        cm_df['Operation'] = op
        cm_df.columns = [['Non-word', 'Real word', 'Operation']]

        dfs.append(cm_df)
    return pd.concat(dfs, axis=0)

def build_example_confusion_matrices(n_ops, n_errors_per_example, ops=OPS, subsetter=lambda df: df):
    dfs = []
    for op in ops:
        df = load_data(op, n_ops, n_errors_per_example)
        df = subsetter(df)
        ecm = build_example_confusion_matrix(df)
        ecm_df = pd.DataFrame(data=ecm)
        ecm_df['op'] = op
        dfs.append(ecm_df)
    return pd.concat(dfs, axis=0)

def build_classification_reports(n_ops, n_errors_per_example, ops=OPS, subsetter=lambda df: df):
    reports = {}
    for op in ops:
        df = load_data(op, n_ops, n_errors_per_example)
        df = subsetter(df)
        reports[op] = build_classification_report(df)
    return reports

"""
def build_analyses(n_ops, n_errors_per_example):
    analyses = collections.defaultdict(dict)
    for op in ['transpose', 'substitute', 'insert', 'delete']:
        confusion_matrix, classification_report, example_confusion_matrix \
                = build_analysis(op, n_ops, n_errors_per_example)
        analyses['confusion_matrix'][op] = confusion_matrix
        analyses['classification_report'][op] = classification_report
        analyses['example_confusion_matrix'][op] = example_confusion_matrix
    return analyses
"""
