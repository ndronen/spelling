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

class LanguageModelClassifier(object):
    def __init__(self, estimators):
        self.estimators = estimators

    def predict_proba(self, X, y=None, key='log_probs', normalize=True):
        proba = np.zeros((len(X), len(self.estimators)))
        for i,estimator in enumerate(self.estimators):
            proba[:, i] = estimator.predict_proba(X, key=key)
        if normalize:
            # Make each row sum to 1.
            proba = proba / proba.sum(axis=1, keepdims=True)
            if key != 'log_probs':
                proba = 1 - proba
                if normalize and len(self.estimators) != 2:
                    # Subtracting from 1 only keeps the sum to 1
                    # if there are 2 estimators.
                    proba = proba / proba.sum(axis=1, keepdims=True)

        return proba

    def predict(self, X, y=None, return_proba=False):
        proba = self.predict_proba(X, y)
        pred = np.argmax(proba, axis=1)
        if return_proba:
            return pred, proba
        else:
            return pred

class CharacterLanguageModel(object):
    """
    Defaults to Witten-Bell discounting for character language models.
    """
    def __init__(self, discount, order=None, model_path=None, keep_model=True, debug=False, encoding='utf8'):
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

    def build_generate_cmd(self, model_path, order, n):
        cmd = [
                self.ngram,
                '-lm', model_path,
                '-order', str(order),
                '-gen', str(n)
                ]
        return cmd


    def write_data(self, f, data):
        for line in data:
            f.write(u'<s> ')
            for char in line:
                f.write(char)
                f.write(u' ')
            f.write(u' </s>\n')
        f.flush()

    def check_X(self, X, caller):
        try:
            if os.path.exists(X):
                pass
            else:
                raise ValueError(("'X' argument of %s must be on-disc file or list. " +
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
            with NamedTemporaryFile(delete=not self.debug, mode='w+') as f:
                train_path = f.name
                if isinstance(X, str):
                    train_path = X
                else:
                    self.write_data(f, X)
                # Run command and raise exception if error.
                fit_cmd = self.build_fit_cmd(train_path, self.model_path)
                try:
                    output = subprocess.check_output(fit_cmd,
                            stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError as cp:
                    lines = cp.output.decode(self.encoding).split('\n')
                    raise RuntimeError("command %s failed: %s",
                        (' '.join(fit_cmd), '\n'.join(lines)))

    def predict(self, X, y=None):
        self.check_X(X, "predict")

        if self.model_path is None:
            if self.Xtrain is None:
                raise ValueError("call fit before calling predict")

            with NamedTemporaryFile(delete=not self.keep_model) as model_file:
                self.model_path = model_file.name
                with NamedTemporaryFile(delete=not self.debug, mode='w+') as train_file:
                    train_path = train_file.name
                    if isinstance(self.Xtrain, str):
                        train_path = self.Xtrain
                    else:
                        # Write self.Xtrain to train_path
                        self.write_data(train_file, self.Xtrain)
                    with NamedTemporaryFile(delete=not self.debug, mode='w+') as test_file:
                        test_path = test_file.name
                        if isinstance(X, str):
                            test_path = X
                        else:
                            # Write X to test_path
                            self.write_data(test_file, X)

                        fit_cmd = self.build_fit_cmd(
                                train_path, self.model_path)
                        if self.debug:
                            print('FIT')
                            print(fit_cmd)
                        try:
                            fit_output = subprocess.check_output(fit_cmd,
                                    stderr=subprocess.STDOUT)
                        except subprocess.CalledProcessError as cp:
                            lines = cp.output.decode(self.encoding).split('\n')
                            raise RuntimeError("command %s failed: %s",
                                (' '.join(fit_cmd), '\n'.join(lines)))
                        if self.debug:
                            print(fit_output)
                        predict_cmd = self.build_predict_cmd(
                                test_path, self.model_path)
                        if self.debug:
                            print('PREDICT')
                            print(predict_cmd)
                        try:
                            predict_output = subprocess.check_output(predict_cmd,
                                    stderr=subprocess.STDOUT)
                        except subprocess.CalledProcessError as cp:
                            lines = cp.output.decode(self.encoding).split('\n')
                            raise RuntimeError("command %s failed: %s",
                                (' '.join(fit_cmd), '\n'.join(lines)))
        else:
            with NamedTemporaryFile(delete=not self.debug, mode='w+') as test_file:
                test_path = test_file.name
                if isinstance(X, str):
                    test_path = X
                else:
                    # Write X to test_path
                    self.write_data(test_file, X)
                predict_cmd = self.build_predict_cmd(
                        test_path, self.model_path)
                if self.debug:
                    print('PREDICT')
                    print(predict_cmd)
                predict_output = subprocess.check_output(predict_cmd,
                        stderr=subprocess.STDOUT)

        return self.get_scores(predict_output)

    def predict_proba(self, X, key='log_probs'):
        pred = self.predict(X)[key]
        if key == 'log_probs':
            return np.exp(pred)
        elif key.startswith('ppl'):
            return pred
        
    def get_fields(self, output, score, column):
        for line in output.decode(self.encoding).split('\n'):
            fields = line.split(' ')
            if score in fields:
                yield fields[column]

    def get_log_probs(self, output):
        log_probs = []
        for log_prob in self.get_fields(output, "logprob=", 3):
            try:
                log_probs.append(float(log_prob))
            except ValueError:
                log_probs.append(0.0)
        return log_probs[:-1]

    def get_ppls(self, output):
        ppls = []
        for ppl in self.get_fields(output, "ppl=", 5):
            try:
                ppls.append(float(ppl))
            except ValueError:
                ppls.append(0.0)
        return ppls[:-1]

    def get_ppl1s(self, output):
        ppl1s = []
        for ppl1 in self.get_fields(output, "ppl1=", 7):
            try:
                ppl1s.append(float(ppl1))
            except ValueError:
                ppl1s.append(0.0)
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
        if self.model_path is None:
            self.predict([""])

        def run():
    		# Generate more examples than requested, as some
    		# of them will be the empty string, and we guarantee
    		# that all generated examples will be non-empty.
            m = max(10, int(0.5 * n))
            generate_cmd = self.build_generate_cmd(self.model_path, order, m)
            if self.debug:
                print(generate_cmd)
                print([type(arg) for arg in generate_cmd])
            try:
                output = subprocess.check_output(generate_cmd,
                    stderr=subprocess.STDOUT)
                words = output.decode(self.encoding).split('\n')
                words = [w.replace(' ', '') for w in words]
                words = [w for w in words if len(w) > 0]
                return words
            except subprocess.CalledProcessError as e:
                raise RuntimeError("Error calling command (%s): %s" %
                        (' '.join(generate_cmd), e.output))

        generated = []
        while len(generated) < n:
            generated.extend(run())
        assert len(generated) >= n
        return generated[0:n]
            
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
