import spelling.mitton
import codecs
import cPickle

def corpus_name(corpus_file):
    name = corpus_file.replace('data/', '')
    name = name.replace('.dat', '')
    name = name.replace('-missp', '')
    return name.title()

def datasets_file(corpus_file):
    name = corpus_name(corpus_file)
    return codecs.open('data/' + name + '.csv', 'w', encoding='utf8')

def evaluation_file(corpus_file):
    name = corpus_name(corpus_file)
    return codecs.open('data/' + name + '-evaluation.csv', 'w', encoding='utf8')

def build_datasets(corpus_file):
    return spelling.mitton.build_mitton_datasets(corpus_file)

def evaluate(dfs):
    return spelling.mitton.evaluate_ranks(dfs)

def run_evaluations():
    for corpus in spelling.mitton.CORPORA:
        print(corpus)
        dfs = build_datasets(corpus)
        name = corpus_name(corpus)
        cPickle.dump(dfs, datasets_file(corpus))
        evaluation = evaluate(dfs)
        evaluation['Corpus'] = name
        cPickle.dump(evaluation, evaluation_file(corpus))
