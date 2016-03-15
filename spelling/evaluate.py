import spelling.mitton
import codecs
import sys
if sys.version_info.major == 2:
    import cPickle as pickle
else:
    import pickle

def corpus_name(corpus_file):
    name = corpus_file.replace('data/', '')
    name = name.replace('.dat', '')
    name = name.replace('-missp', '')
    return name.title()

def datasets_file(corpus_file):
    name = corpus_name(corpus_file)
    return codecs.open('data/' + name + '.pkl', 'w', encoding='utf8')

def evaluation_file(corpus_file):
    name = corpus_name(corpus_file)
    return codecs.open('data/' + name + '-evaluation.pkl', 'w', encoding='utf8')

def build_datasets(corpus_file):
    return spelling.mitton.build_mitton_datasets(corpus_file)

def evaluate(dfs):
    return spelling.mitton.evaluate_ranks(dfs)

def run_evaluations():
    for corpus in spelling.mitton.CORPORA:
        print(corpus)
        dfs = build_datasets(corpus)
        name = corpus_name(corpus)
        #pickle.dump(dfs, datasets_file(corpus), encoding='utf8')
        evaluation = evaluate(dfs)
        yield name, dfs, evaluation
        print(evaluation)
        #evaluation['Corpus'] = name
        #pickle.dump(evaluation, evaluation_file(corpus), encoding='utf8')
