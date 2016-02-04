import sys, os
import numpy as np
import pandas as pd
import argparse
import subprocess
from tempfile import NamedTemporaryFile

DISCOUNTS = {
        'witten-bell': 'wbdiscount',
        'kneser-ney': 'kndiscount'
        }

class CharacterLanguageModel(object):
    """
    Defaults to Witten-Bell discounting for character language models.
    """
    def __init__(self, discount, order=None, model_path=None):
        if discount != 'witten-bell':
            if order is None:
                raise ValueError('"order" is required with %s discounting' % discount)

        if discount not in DISCOUNTS:
            raise ValueError('invalid discount "%s"; valid discunts are %s' %
                    (discount, ','.join(DISCOUNTS.keys())))

        self.__dict__.update(locals())
        del self.self

        self.ngram_count = 'ngram-count'
        self.ngram = 'ngram'
        # For caching training data across fit/predict calls.
        self.Xtrain = None 

    def build_fit_cmd(self, data_path, model_path):
        cmd = [
                self.ngram_count,
                '-lm', model_path,
                '-%s' % DISCOUNTS[self.discount],
                '-text', data_path,
                '-debug', '1'
            ]
        return cmd

    def build_predict_cmd(self, data_path, model_path):
        cmd = [
                self.ngram,
                '-lm', model_path,
                '-ppl', data_path,
                '-debug', '1'
                ]
        return cmd

    def write_data(self, f, data):
        for line in data:
            f.write(u'<s> ')
            for char in line:
                f.write(char)
                f.write(u' ')
            f.write(u' </s>\n')

    def check_X(self, X, caller):
        try:
            if os.path.exists(X):
                pass
            else:
                raise ValueError(("'X' argument of %s must be on-disc file or data. " +
                    "%s is string but path does not exist") % (caller, X))
        except TypeError:
            pass

    def fit(self, X, y=None):
        self.check_X(X, "fit")

        if self.model_path is None:
            # We can't save the model to disc, so don't train until
            # we're given a test set.
            self.Xtrain = X
        else:
            with NamedTemporaryFile() as f:
                train_path = f.name
                if isinstance(X, str):
                    train_path = X
                else:
                    self.write_data(f, X)
                # Run command and raise exception if error.
                fit_cmd = self.build_fit_cmd(train_path, self.model_path)
                try:
                    subprocess.check_call(fit_cmd)
                except OSError as e:
                    raise ValueError("command %s failed: %s",
                            (' '.join(fit_cmd), str(e)))

    def predict(self, X, y=None):
        self.check_X(X, "predict")

        if self.model_path is None:
            if self.Xtrain is None:
                raise ValueError("call fit before calling predict")

            with NamedTemporaryFile() as model_file:
                model_path = model_file.name
                with NamedTemporaryFile() as train_file:
                    train_path = train_file.name
                    if isinstance(self.Xtrain, str):
                        train_path = self.Xtrain
                    else:
                        # Write self.Xtrain to train_path
                        self.write_data(train_file, self.Xtrain)
                    with NamedTemporaryFile() as test_file:
                        test_path = test_file.name
                        if isinstance(X, str):
                            test_path = X
                        else:
                            # Write X to test_path
                            self.write_data(test_file, X)

                        fit_cmd = self.build_fit_cmd(
                                train_path, model_path)
                        subprocess.check_call(fit_cmd)
                        predict_cmd = self.build_predict_cmd(
                                test_path, model_path)
                        output = subprocess.check_output(predict_cmd)
        else:
            with NamedTemporaryFile() as test_file:
                test_path = test_file.name
                if isinstance(X, str):
                    test_path = X
                else:
                    # Write X to test_path
                    self.write_data(test_file, X)
                predict_cmd = self.build_predict_cmd(
                        test_path, self.model_path)
                print('predict_cmd', predict_cmd)
                output = subprocess.check_output(predict_cmd)

        return self.get_scores(output)

    def get_fields(self, output, score, column):
        for line in output.split('\n'):
            fields = line.split(' ')
            if score in fields:
                yield fields[column]

    def get_log_probs(self, output):
        log_probs = []
        for log_prob in self.get_fields(output, "logprob=", 3):
            log_probs.append(log_prob)
        return log_probs[:-1]

    def get_ppls(self, output):
        ppls = []
        for ppl in self.get_fields(output, "ppl=", 4):
            ppls.append(ppl)
        return ppls[:-1]

    def get_ppl1s(self, output):
        ppl1s = []
        for ppl1 in self.get_fields(output, "ppl1=", 4):
            ppl1s.append(ppl1)
        return ppl1s[:-1]

    def get_scores(self, output):
        log_probs = self.get_log_probs(output)
        ppls = self.get_ppls(output)
        ppl1s = self.get_ppl1s(output)
        return {
                'log_probs': log_probs,
                'ppls': ppls,
                'ppl1s': ppl1s
                }

    def generate(self, order, n):
        generate_cmd = ['ngram', '-order', str(order), '-lm', 'char.lm', '-gen', str(n)]
        output = subprocess.check_output(generate_cmd, stderr=subprocess.STDOUT)
        return output.split('\n')
            
def main(args):
    lm = CharacterLanguageModel(args.order, args.model_path)
    lm.fit(args.train_file)
    output = lm.predict(args.test_file)
    log_probs = lm.get_log_probs(output)
    ppls = lm.get_ppls(output)
    ppl1s = lm.get_ppl1s(output)

def build_parser():
    parser = argparse.ArgumentParser(
        description='train and evaluate a language model on positive and negative examples')
    parser.add_argument(
        'train_file', metavar='TRAIN_FILE', type=str,
        help='file containing training examples')
    parser.add_argument(
        'test_file', metavar='TEST_FILE', type=str,
        help='file containing test examples')
    parser.add_argument(
        '--order', type=int, default=3,
        help='the N of the N-gram model')
    parser.add_argument(
        '--model-path', type=str,
        help='the path of the model file; the model is saved here after training')
    return parser.parse_args()

if __name__ == '__main__':
    sys.exit(main(build_parser()))
