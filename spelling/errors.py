import codecs
from langdetect import detect
from collections import defaultdict, Counter, namedtuple
import re
from tqdm import tqdm
from pyxdameraulevenshtein import damerau_levenshtein_distance as dl_distance

def ngram_generator(text, n):
    text = "-"*(n-1) + text
    for i in xrange(len(text)-n):
        yield text[i:i+n]

def tokenize(text):
    return re.findall("[a-z']+", text.lower()) 

def get_edit_function(incorrect, correct):
    for index,(c,h) in enumerate(zip(correct,incorrect)):
        if c != h:
            break
    else:
        index = len(correct)
    if len(incorrect) > len(correct):
        return lambda x,i : x[:i] + incorrect[index] + x[i:], index
    if len(incorrect) < len(correct):
        return lambda x,i : x[:i] + x[i+1:], index
    if index+1 >= len(correct) or correct[index+1] == incorrect[index+1]:
        #mutation
        return lambda x,i : x[:i] + incorrect[index] + x[i+1:], index
    return lambda x,i : x[:i] + x[i+1] + x[i] + x[i+2:], index

def error_iterator(self, fn, dictionary, whitelist=None):
    #TODO: utf-8 shouldn't be hardcoded
    if whitelist is None:
        whitelist = set()
    num_correct = 0
    num_incorrect = 0
    whitelist = set()
    with codecs.open(fn, "r", "utf-8") as f:
        for line in tqdm(f):
            try:
                lang = detect(line)
            except Exception:
                continue
            if lang != "en":
                continue
            for word in tokenize(line.lower()):
                if word[-2:] == "'s":
                    word = word[:-2]
                if not word:
                    continue
                if word[0] == "'":
                    word = word[1:]
                if not word:
                    continue
                if word[-1] == "'":
                    word = word[:-1]
                if word in whitelist or dictionary.check(word):
                    continue
                corrected = dictionary.correct(word)
                if corrected == word:
                    num_correct += 1
                elif corrected:
                    num_incorrect += 1
                    yield word, corrected

class ErrorGenerator(object):

    def __init__(self, source_fn, dictionary, ngram_size=3):
        self.generators = defaultdict(list)
        self.statistics = Counter()
        self.errors_seen = 0
        self.global_error_rate = 0
        self.ngram_size = ngram_size
        self._train(source_fn, dictionary)

    def _train(self, fn, dictionary):
        for incorrect, correct in self._extract_errors(fn, dictionary):
            if dl_distance(incorrect, correct) == 1:
                edit_function, index = get_edit_function(incorrect, correct)
                padded = "-"*(self.ngram_size-1) + correct
                index += self.ngram_size-1
                trigram = padded[index-2:index+1]
                self.generators[trigram].append(edit_function)

    def generate(self, word):
        Error = namedtuple("Error", ["correct","incorrect","error_name","probability"])
        results = []
        for i,ngram in enumerate(self.ngram_generator(word)):
            gens = self.generators[ngram]
            for gen,weight in gens.iteritems():
                corrupted = gen.injector(word, i)
                prob = weight/float(self.errors_seen)
                results.append(Error(correct=word, incorrect=corrupted, 
                                      error_name=gen.name, probability=prob))
        return results
