import codecs
import pandas as pd
from spelling.features import levenshtein_distance as dist
from spelling.jobs import DistanceToNearestStem

df = pd.read_csv('data/aspell-dict.csv.gz', sep='\t', encoding='utf8')
job = DistanceToNearestStem()
df = job.run(df.word, dist)
# TODO: Merge the two aspell-dict files.
pd.to_csv('data/aspell-dict-distances.csv.gz', index=False, sep='\t',
        encoding='utf8', compression='gz')
