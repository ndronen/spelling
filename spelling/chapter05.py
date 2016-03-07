# coding: utf-8

import collections
import pandas as pd

import spelling.mitton
import spelling.dictionary
from spelling.features import distance as spelldist

from spelling.utils import build_progressbar as progressbar

def build_dictionaries(vocabulary):
    sorter = spelling.dictionary.DistanceSorter('jaro_winkler')

    aspell = spelling.dictionary.AspellRetriever()
    aspell_rtr = spelling.dictionary.SortingRetriever(aspell, sorter)

    ed = spelling.dictionary.EditDistanceRetriever(vocabulary)
    ed_rtr = spelling.dictionary.SortingRetriever(ed, sorter)

    return { 'aspell': aspell_rtr, 'edit_distance': ed_rtr }

def load_error_correction_pairs(path):
    words = spelling.mitton.load_mitton_words(path)
    return spelling.mitton.build_mitton_pairs(words)

def add_edit_distances_from_error_to_correction(df, distances=['jaro_winkler', 'levenshtein_distance']):
    for d in distances:
        func = lambda row: spelldist(row['real_word'], row['non_word'], d)
        df[d] = df.apply(func, axis=1)

def run():
    vocab_df = pd.read_csv('data/aspell-dict.csv.gz', sep='\t', encoding='utf8')
    dicts = build_dictionaries(vocab_df.word.tolist())
    
    files = [
        'data/aspell.dat', 
        'data/holbrook-missp.dat',
        'data/norvig.dat',
        'data/wikipedia.dat',
        'data/birbeck.dat']

    for f in files:
        corpus_name = f.replace('data/', '').replace('.dat', '')
        print(corpus_name)

        pairs = load_error_correction_pairs(f)
        df = pd.DataFrame(data={
                'non_word': [p[0] for p in pairs],
                'real_word': [p[1] for p in pairs]
                })
        add_edit_distances_from_error_to_correction(df)
        df['corpus'] = corpus_name
        
        for d in dicts.keys():
            pbar = progressbar(pairs)
            rtr = dicts[d]
            ranks = []
            for i,pair in enumerate(pairs):
                pbar.update(i+1)
                error, correction = pair
                try:
                    ranks.append(rtr[error].index(correction))
                except ValueError:
                    ranks.append(-1)
            pbar.finish()
            colname = d + '_rank'
            df[colname] = ranks

        yield df

def build_rank_frequency_table(df):
    columns = [col for col in df.columns if '_rank' in col]
    max_rank = max(df[columns].max())

    rank_count = collections.defaultdict(list)
    for i in range(-1, max_rank+1):
        if i > -1:
            desc = "%d" % (i+1)
        else:
            desc = None
        rank_count['position_in_list'].append(desc)

    for col in columns:
        counts = df[col].value_counts().sort_index()
        for i in range(-1, max_rank+1):
            c = counts[counts.index == i].values
            count = 0 if len(c) == 0 else c[0]
            rank_count[col].append(count)

    return pd.DataFrame(data=rank_count)
