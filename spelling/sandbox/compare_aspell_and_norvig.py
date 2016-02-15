reload(spelling.mitton)
reload(spelling.dictionary)

from spelling.mitton import build_mitton_datasets, evaluate_ranks
from spelling.dictionary import (AspellWithGoogleLanguageModel,
        NorvigWithAspellDictAndGoogleLanguageModel)

CONSTRUCTORS = [
            AspellWithGoogleLanguageModel,
            NorvigWithAspellDictAndGoogleLanguageModel
            ]

def build_datasets(constructors=CONSTRUCTORS):
    dfs = build_mitton_datasets('data/birbeck.dat',
        constructors=CONSTRUCTORS)

def plot_performance(dfs, ranks, ignore_case=False):
    df = evaluate_ranks(dfs, ranks, ignore_case=ignore_case)
    print(df)

def run(dfs=None, ranks=None, ignore_case=False, constructors=CONSTRUCTORS):
    if dfs is None:
        dfs = build_datasets(constructors)
    if ranks is None:
        ranks = np.arange(10) + 1
    plot_performance(dfs, ranks, ignore_case=ignore_case)

    # Let rank = 10.
    # For each dictionary's suggestions, 
