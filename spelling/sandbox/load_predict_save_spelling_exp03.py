import os
import h5py
import itertools
import modeling.utils
import numpy as np
import pandas as pd

from sklearn.metrics import (confusion_matrix, classification_report)

BASE_DIR = os.environ['HOME'] + '/proj/modeling/'
MODEL_DIR = BASE_DIR + '/models/keras/spelling/convnet/exp03-inputs/'
DATA_DIR = BASE_DIR + '/data/spelling/experimental/'

MODEL_TEMPLATE = 'op_%s_n_ops_%d_n_errors_per_word_%d'
DATA_TEMPLATE = 'op-%s-distance-%d-errors-per-word-%d'

def build_csv_path(op, n_ops, n_errors_per_op):
    return DATA_DIR + (DATA_TEMPLATE % (op, n_ops, n_errors_per_op)) + '.csv'

def build_pred_path(op, n_ops, n_errors_per_op):
    return DATA_DIR + (DATA_TEMPLATE % (op, n_ops, n_errors_per_op)) + '-exp03-inputs-pred.h5'

def build_model_cfg_path(model_op, model_n_ops, model_n_errors_per_op, data_op, data_n_ops, data_n_errors_per_op):
    model_cfg_template = 'data/spelling/experimental/op-%s-distance-%d-errors-per-word-%d-model-%s-%d-%d-cfg.json'
    return model_cfg_template % (data_op, data_n_ops, data_n_errors_per_op,
            model_op, model_n_ops, model_n_errors_per_op)

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


def load_data(op, n_ops, n_errors_per_op):
    raise ValueError("Change this to take data and model arguments.")
    csv_path = build_csv_path(op, n_ops, n_errors_per_op)
    pred_path = build_pred_path(op, n_ops, n_errors_per_op)

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

def build_confusion_matrix(op, n_ops, n_errors_per_op, df=None):
    if df is None:
        df = load_data(op, n_ops, n_errors_per_opt)
    return confusion_matrix(df.binary_target, df.pred)

def build_classification_report(op, n_ops, n_errors_per_op, df=None, digits=4):
    if df is None:
        df = load_data(op, n_ops, n_errors_per_opt)
    return classification_report(df.binary_target, df.pred, digits=digits)

def build_example_confusion_matrix(op, n_ops, n_errors_per_op, df=None):
    if df is None:
        df = load_data(op, n_ops, n_errors_per_opt)

    def get_word_and_pr(df, target, right=True, ascending=True, n=10):
        prob_name = 'prob%d' % target
        mask = df.pred == target
        if not right:
            mask = ~mask
        df = df[mask][df.binary_target == target]
        return df.sort_values(prob_name, ascending=ascending).head(n)

    targets = [0, 1]
    model_is_right = [True, False]
    ascending = [True, False]
    dfs = []
    for tup in itertools.product(targets, model_is_right, ascending):
        target = tup[0]
        right = tup[1]
        ascending = tup[2]
        prob_name = 'prob%d' % target
        df_tmp = get_word_and_pr(df, target, right=right, ascending=ascending)
        df_tmp = df_tmp[['word', 'real_word', 'binary_target', 'pred', 'prob0', 'prob1']]
        #print(target, right, ascending)
        #print(df_tmp)
        dfs.append(df_tmp)

    return pd.concat(dfs)

def build_report(op, n_ops, n_errors_per_op):
    raise ValueError("Change this to take data and model arguments.")
    df = load_data(op, n_ops, n_errors_per_op)
    return build_confusion_matrix(op, n_ops, n_errors_per_op, df=df), \
        build_classification_report(op, n_ops, n_errors_per_op, df=df), \
        build_example_confusion_matrix(op, n_ops, n_errors_per_op, df=df)
