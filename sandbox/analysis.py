import spelling.mitton
from spelling.mitton import CORPORA
from spelling.features import keyboard_distance, levenshtein_distance
import numpy as np
import pandas as pd

corpus_names = [c.replace('data/', '') for c in CORPORA]
fractions = []

for corpus in CORPORA:
    words = spelling.mitton.load_mitton_words(corpus)
    pairs = spelling.mitton.build_mitton_pairs(words)
    pairs = [(p[0], p[1],
        levenshtein_distance(p[0], p[1]),
        keyboard_distance(p[0], p[1])) for p in pairs]
    df = pd.DataFrame(data=pairs, columns=[
            'word', 'correct_word', 'edit_distance', 'keyboard_distance'])
    r = np.corrcoef(df.edit_distance, df.keyboard_distance)[0][1]
    edist_table = df.edit_distance.value_counts().values
    fractions.append((edist_table / float(edist_table.sum()))[0:3])

fraction_df = pd.DataFrame(data=fractions, columns=["1", "2", "3"])
fraction_df['Corpus'] = corpus_names
fraction_df = fraction_df[["Corpus", "1", "2", "3"]]
print(fraction_df.to_latex(float_format=lambda f: '%3d' % 100*f))
