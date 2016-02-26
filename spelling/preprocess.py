import numpy as np
import pandas as pd
import h5py
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

def build_vocab(words, zero_character='|', lowercase=False, analyzer='char_wb', **kwargs):
    vocab = CountVectorizer(analyzer=analyzer, lowercase=lowercase, **kwargs)
    vocab.fit(words)
    index_to_char = dict([(v,k) for k,v in vocab.vocabulary_.iteritems()])
    tmp_v = index_to_char[0]
    index_to_char[len(index_to_char)] = index_to_char[0]
    index_to_char[0] = zero_character
    char_to_index = dict([(v,k) for k,v in index_to_char.iteritems()])
    return char_to_index

def build_ascii_vocab(**kwargs):
    chars = [chr(_) for _ in range(255)]
    idx = range(255)
    return dict(zip(chars, idx))

"""
def add_to_vocab(char_to_index, chars):
    char_to_index = dict(char_to_index)
    for char in chars:
        char_to_index[char] = len(char_to_index)
    return char_to_index
"""

def load_dictionary_words(path='data/aspell-dict.csv.gz'):
    df = pd.read_csv(path, sep='\t', encoding='utf8')
    return df.word

def build_nonce_chars(nonce_interval, longest_word):
    nonce_chars = []
    if nonce_interval > 0:
        n_nonces = int(np.ceil(longest_word/float(nonce_interval)))
        nonce_chars = [chr(_) for _ in range(n_nonces)]
    return nonce_chars

def add_nonces_to_word(word, nonce_chars, nonce_interval):
    if nonce_interval < 1:
        return word

    n_nonces = len(nonce_chars)
    # Split the word into nonce_interval-1 sized subsequences.
    split = [word[j:j+nonce_interval] for j in range(0, len(word), nonce_interval)]
    accum = []
    # Append a unique nonce character to each split.
    for k,s in enumerate(split):
        if k < len(split)-1:
            s = s + nonce_chars[k]
        accum.append(s)
    word = ''.join(accum)

    return word

def build_char_matrix(words, width=25, nonce_interval=0):
    """
    Parameters
    ----------
    words : list of str or unicode
        List of (spelling error, correction) pairs.
    width : int
        The width of the matrix.  Words longer than
    nonce_interval : int 
        The distance between nonce characters; 0 disables them.
        Maximum nonce characters is 10, so when using nonces,
        words longer than 10*nonce_interval are discarded.
    """
    try:
        longest_word = max([len(w) for w in words])
    except ValueError as e:
        print(e, words)

    if nonce_interval > 0:
        assert longest_word < 10 * nonce_interval

    nonce_chars = []
    if nonce_interval > 0:
        nonce_chars = build_nonce_chars(nonce_interval, longest_word)

    char_to_index = build_ascii_vocab()

    # Exclude words that are too long.
    mask = np.zeros(len(words)).astype(bool)
    for i,word in enumerate(words):
        word_nonce = add_nonces_to_word(word, nonce_chars, nonce_interval)
        if len(word_nonce) > width:
            print('word (%s) does not fit in the matrix (width %d)' % (word, width))
            continue
        mask[i] = True

    # Build the matrix.
    char_matrix = np.zeros((len(words), width))
    for i,word in enumerate(words):
        word_nonce = add_nonces_to_word(word, nonce_chars, nonce_interval)
        for k,char in enumerate(word_nonce):
            try:
                char_matrix[i, k] = char_to_index[char]
            except IndexError:
                pass

    return char_matrix, mask

def build_char_matrix_split(pairs):
    raise NotImplementedError()
    data, target = build_char_matrix(pairs)
    return train_test_split(data, target, train_size=0.9)

def build_hdf5_file(path, data_name, data, target_name, target):
    f = h5py.File(path, 'w')
    f.create_dataset(data_name, data=data, dtype=np.int32)
    f.create_dataset(target_name, data=target, dtype=np.int32)
    f.close()
