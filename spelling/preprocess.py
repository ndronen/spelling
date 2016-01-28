import pandas as pd
import h5py
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import spelling.mitton

def build_vocab(words, zero_character='|', lowercase=False, analyzer='char_wb', **kwargs):
    vocab = CountVectorizer(analyzer=analyzer, lowercase=lowercase, **kwargs)
    vocab.fit(words)
    index_to_char = dict([(v,k) for k,v in vocab.vocabulary_.iteritems()])
    tmp_v = index_to_char[0]
    index_to_char[len(index_to_char)] = index_to_char[0]
    index_to_char[0] = zero_character
    char_to_index = dict([(v,k) for k,v in index_to_char.iteritems()])
    return char_to_index

def load_dictionary_words(path='data/aspell-dict.csv.gz'):
    df = pd.read_csv(path, sep='\t', encoding='utf8')
    return df.word

def load_pairs(path):
    words = spelling.mitton.load_mitton_words(path)
    pairs = spelling.mitton.build_mitton_pairs(words)
    return pairs

def load_non_words(path):
    # Non-words are in the first position.
    pairs = load_pairs(path)
    return [t[0] for t in pairs]

def load_real_words(path):
    # Real words are in the second position.
    pairs = load_pairs(path)
    return [t[1] for t in pairs]

def build_data_target(pairs):
    max_len = max([max(len(t[0]), len(t[1])) for t in pairs])

    words = {}
    for pair in pairs:
        # Nonword.
        words[pair[0]] = 0
        # Real word.
        words[pair[1]] = 1

    char_to_index = build_vocab(words.iterkeys(), ngram_range=(1,1))

    data = np.zeros((len(words), max_len))
    target = np.zeros(len(words))
    for i,word in enumerate(words.keys()):
        for j,char in enumerate(word):
            data[i, j] = char_to_index[char]
        target[i] = words[word]
    return data, target

def build_data_target_split(pairs):
    data, target = build_data_target(pairs)
    return train_test_split(data, target, train_size=0.9)

def build_hdf5_file(path, data_name, data, target_name, target):
    f = h5py.File(path, 'w')
    f.create_dataset(data_name, data=data, dtype=np.int32)
    f.create_dataset(target_name, data=target, dtype=np.int32)
    f.close()

def count_vocabulary(words, **kwargs):
    return(len(build_vocab(words, **kwargs)))

def run_corpus(path, max_ngram_range=6):
    sizes = {}
    sizes['realword'] = {}
    sizes['nonword'] = {}

    real_words = load_real_words(path)
    for k in range(1, max_ngram_range):
        sizes['realword'][k] = count_vocabulary(real_words, ngram_range=(k,k))

    non_words = load_non_words(path)
    for k in range(1, 6):
        sizes['nonword'][k] = count_vocabulary(non_words, ngram_range=(k,k))

    print(sizes)

def run_dictionary(path='data/aspell-dict.csv.gz', max_ngram_range=6):
    sizes = {}
    words = load_dictionary_words(path)
    for k in range(1, max_ngram_range):
        sizes[k] = count_vocabulary(words, ngram_range=(k,k))
    print(sizes)
