#!/usr/bin/env python

import inspect
path = inspect.getfile(inspect.currentframe())
from os.path import dirname, split
import sys
sys.path.append(dirname(split(path)[0]))

import argparse
from spelling.jobs import ErrorExtractionJob
from spelling import dictionary
import pandas
from pandas import DataFrame
import collections

def main(dictionary, words_fn, corpus_fn, whitelist_fns):
    #d = dictionary.AspellUniword()
    #social_fn = "/Users/uhellsc/data/socialmedia.txt"
    #dictionary_fn = "/usr/share/dict/words"
    #corpus_fn = "/Users/uhellsc/data/graph/responseText-500k.txt"
    #corpus_fn = "/Users/uhellsc/data/graph/truncated.txt"

    #words = ["this", "is", "a", "test", "of", "the", "program", "and", "some", "other", "words", "back"]
    #aspell_data = pandas.read_csv("data/aspell-dict.csv", sep="\t")
    words_dataframe = pandas.read_csv(words_fn, sep="\t")
    words = words_dataframe["word"].tolist()

    job = ErrorExtractionJob(words, dictionary, corpus_fn, whitelist_fns)
    result = job.run()
    result_dict = collections.defaultdict(list)
    for entry in result:
        result_dict["word"].append(entry.correct)
        result_dict["error"].append(entry.incorrect)
        result_dict["probability"].append(entry.probability)
        result_dict["error_name"].append(entry.error_name)

    df = DataFrame(data=result_dict)
    print df.head()
    df.to_csv("out.csv", sep='\t')

def build_parser():
    parser = argparse.ArgumentParser(
            description="Mine errors from a corpus and then apply those errors to a list of words")
    aa = parser.add_argument
    aa("--corpus", help="Input file containing text to mine for error patterns")
    aa("--words", help="Input file with words to add errors to. Expected to be a csv with a 'word' column")
    #aa("--dictionary", type=str,
    #        default="/pkt-aggregator/Service/ngram",
    #        help="The path component of the URL of the N-gram service")
    aa("--whitelists", type=str, nargs="*",
            help="Text files containing words that should be considered correctly spelled (one per line)")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = build_parser()
    sys.exit(main(dictionary.AspellUniword(), args.words, args.corpus, args.whitelists))
