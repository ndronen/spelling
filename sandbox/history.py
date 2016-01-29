# coding: utf-8

import numpy as np
rng = np.random.RandomState(seed=17)
def jitter(x, scale=0.1):
    return x + rng.normal(scale=scale, size=x.shape) 

import matplotlib.pyplot as plt
import pandas as pd

from spelling.mitton import build_mitton_datasets
from spelling.dictionary import NorvigWithAspellDictAndGoogleLanguageModel as Model

df_dist = pd.read_csv('data/aspell-dict-distances.csv.gz',
        sep='\t', encoding='utf8')

model_df = build_mitton_datasets('data/aspell.dat',
        constructors=[Model])['NorvigWithAspellDictAndGoogleLanguageModel']

correct_suggestions = model_df[(model_df.target == 1)]

y = correct_suggestions.suggestion_index

x2 = correct_suggestions.suggestion_count
error_word_length = np.array([len(w) for w in correct_suggestions.error])
plt.figure()
plt.scatter(jitter(x2), jitter(y), alpha=0.1, c='green', s=10*error_word_length**2)
plt.xlabel("Number of suggestions")
plt.ylabel("Index of correct word in suggestions")
plt.show(block=False)

x2 = correct_suggestions.suggestion_count
y = np.zeros(len(correct_suggestions))
for i,word in enumerate(correct_suggestions.correct_word):
    y[i] = df_dist[df_dist.word == word].distance.values[0]

plt.figure()
plt.scatter(jitter(x2), jitter(y), alpha=0.1, c='green', s=10*error_word_length**2)
plt.xlabel("Number of suggestions")
plt.ylabel("Edit distance from correct word to nearest word in dictionary")
plt.show(block=False)
