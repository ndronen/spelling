from __future__ import print_function

import collections
import cPickle
import json
import h5py
import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split

from spelling.edits import Editor, EditFinder
from spelling.preprocess import build_char_matrix
from spelling.utils import build_progressbar

OPERATIONS = ['delete', 'insert', 'substitute', 'transpose']

def save_corpus_as_hdf5(df, path, nonce_interval, width=25):
    char_matrix, char_mask = build_char_matrix(df.word, width=width, nonce_interval=nonce_interval)
    marked_char_matrix, marked_char_mask = build_char_matrix(df.marked_word, width=width+2, nonce_interval=nonce_interval)
    # Only keep the examples that are in both char_matrix and marked_char_matrix.
    mask = char_mask & marked_char_mask

    f = h5py.File(path, 'w')
    f.create_dataset('chars', data=char_matrix[mask], dtype=np.int32)
    f.create_dataset('marked_chars', data=marked_char_matrix[mask], dtype=np.int32)
    f.create_dataset('binary_target', data=df.binary_target.values[mask], dtype=np.int32)
    f.create_dataset('multiclass_target', data=df.multiclass_target.values[mask], dtype=np.int32)
    f.create_dataset('distance', data=df.distance.values[mask], dtype=np.int32)
    f.close()

def save_corpus_as_csv(df, path):
    df.to_csv(path, sep='\t', encoding='utf8')

def save_target_for_srilm(words, path):
    with open(path, 'w') as f:
        for word in words:
            word = ' '.join(word)
            f.write('<s>')
            for char in word:
                f.write(char)
            f.write('</s>\n')

def save_corpus_for_srilm(df, prefix):
    # Save one file for real words and one for non-words.
    save_target_for_srilm(df[df.binary_target == 1].word,
            prefix + '-real-words.txt')
    save_target_for_srilm(df[df.binary_target == 0].word,
            prefix + '-non-words.txt')

def class_weights(targets):
    n_samples = len(targets)
    n_classes = np.max(targets) + 1.
    class_keys = [str(k) for k in np.sort(np.unique(targets))]
    class_weights = (n_samples / (n_classes * np.bincount(targets))).tolist()
    return dict(zip(class_keys, class_weights))
    
def save_target_data(df, path):
    target_data = {}

    # Compute target weights.
    n_samples = len(df)
    target_data['binary_target'] = {}
    target_data['binary_target']['names'] = ['0', '1']
    target_data['binary_target']['weights'] = class_weights(df.binary_target.values)

    target_data['multiclass_target'] = {}
    target_data['multiclass_target']['names'] = [str(x) for x in sorted(df.multiclass_target)]
    target_data['multiclass_target']['weights'] = class_weights(df.multiclass_target.values)

    cPickle.dump(target_data, open(path + '.pkl', 'w'))
    json.dump(target_data, open(path + '.json', 'w'))

def save_corpus(df, prefix, nonce_interval):
    save_corpus_as_hdf5(df, prefix + '.h5', nonce_interval)
    save_corpus_as_csv(df, prefix + '.csv')
    save_corpus_for_srilm(df, prefix)
    save_target_data(df, prefix)

def build_and_save_corpora(distance, n, nonce_interval=0, operations=OPERATIONS, random_state=17):
    df = build_operation_corpora(distance, n=n, operations=operations)
    df = df.sample(frac=1., random_state=random_state)
    for operation in operations:
        save_corpus(
                df[df.operation == operation],
                'op-%s-distance-%d-errors-per-word-%d' % (operation, distance, n),
                nonce_interval)

def build_real_word_corpus(real_words):
    corpus = init_corpus()

    # Multiclass target starts at 1; 0 is assumed to be the negative class.
    for i,word in enumerate(real_words):
        corpus['word'].append(word)
        corpus['marked_word'].append('^' + word + '$')
        corpus['real_word'].append(word)
        corpus['binary_target'].append(1)
        corpus['multiclass_target'].append(i+1)
        corpus['orig_pattern'].append('')
        corpus['changed_pattern'].append('')
        corpus['distance'] = 0
        corpus['operation'] = ''

    return corpus

def init_corpus():
    corpus = {}
    corpus['word'] = []
    corpus['marked_word'] = []
    corpus['real_word'] = []
    corpus['binary_target'] = []
    corpus['multiclass_target'] = []
    corpus['orig_pattern'] = []
    corpus['changed_pattern'] = []
    return corpus

def add_non_word_to_corpus(corpus, real_word, non_word, orig_pattern='', changed_pattern='', operation='', distance=0):
    corpus['word'].append(non_word)
    corpus['marked_word'].append('^' + non_word + '$')
    corpus['real_word'].append(real_word)
    corpus['binary_target'].append(0)
    corpus['multiclass_target'].append(0)
    corpus['orig_pattern'].append(orig_pattern)
    corpus['changed_pattern'].append(changed_pattern)
    corpus['distance'] = distance
    corpus['operation'] = operation

def build_operation_corpora(distance, words=None, dict_path='data/aspell-dict.csv.gz', operations=OPERATIONS, n=3, random_state=17):
    corpora = collections.defaultdict(list)

    if words is None:
        df = pd.read_csv(dict_path, sep='\t', encoding='utf8')
        words = df.word.tolist()

    for operation in operations:
        print(operation)

        corpus = build_operation_corpus(distance, operation,
                words, n=n, random_state=random_state)

        for k,v in corpus.iteritems():
            corpora[k].extend(v)

    return pd.DataFrame(corpora)

def build_operation_corpus(distance, operation, words, n=3, random_state=17):
    if isinstance(random_state, int):
        random_state = np.random.RandomState(seed=random_state)

    editor = Editor()
    edit_finder = EditFinder()
    pbar = build_progressbar(words)

    corpus = init_corpus()

    words_set = set(words)

    for i,w in enumerate(words):
        pbar.update(i+1)
        edits = set([w])
        #print('initial edits', edits)
        for i in range(distance):
            #print(w, i)
            new_edits = set()
            for edit in edits:
                #print('getting edits for %s' % edit)
                edits_for = editor.edit(edit, operation)
                new_edits.update(edits_for)
                #print('edits for %s %s' % (edit, str(new_edits)))

            # Remove the word itself from new edits.
            try:
                new_edits.remove(w)
            except KeyError:
                pass

            # Remove real words from the edits.
            for edit in new_edits.copy():
                if edit in words_set:
                    new_edits.remove(edit)

            # Break out if we can't make any new edits.
            if len(new_edits) == 0:
                new_edits = edits
                break

            #print('new edits for %s %s (after removing %s)' % (edit, str(new_edits), w))

            n_choice = min(n, len(new_edits))

            try:
                edits = random_state.choice(list(new_edits), size=n_choice, replace=False)
            except ValueError as e:
                #print(w, new_edits, e)
                raise e

            #print('%d edits for %s %s (after sampling %d)' % (n_choice, edit, str(edits), n))

        try:
            edits = random_state.choice(list(edits), size=n, replace=False)
        except ValueError:
            pass

        for edit in edits:
            corpus['word'].append(unicode(edit))
            # Use start-of-word and end-of-word markers as in http://arxiv.org/abs/1602.02410.
            corpus['marked_word'].append('^' + edit + '$')
            corpus['real_word'].append(w)
            corpus['binary_target'].append(0)
            corpus['multiclass_target'].append(0)

            orig_chars = []
            changed_chars = []
            for orig,changed in edit_finder.find(w, edit):
                orig_chars.append(orig)
                changed_chars.append(changed)
            corpus['orig_pattern'].append('-'.join(orig_chars))
            corpus['changed_pattern'].append('-'.join(changed_chars))

    pbar.finish()

    corpus['distance'] = [distance for w in corpus['word']]
    corpus['operation'] = [operation for w in corpus['word']]

    return corpus

