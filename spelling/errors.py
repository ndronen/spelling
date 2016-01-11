import codecs
from langdetect import detect
from collections import defaultdict, Counter, namedtuple
import re
from tqdm import tqdm
from pyxdameraulevenshtein import damerau_levenshtein_distance as dl_distance

Error = namedtuple("Error", ["correct","incorrect","error_name","probability"])
Injector = namedtuple("Injector", ["function","name"])

class HashableInjector(Injector):
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        return self.name == other.name

def ngram_generator(text, n):
    text = "-"*(n-1) + text
    for i in xrange(len(text)-n):
        yield text[i:i+n]

def tokenize(text):
    return re.findall("[a-z']+", text.lower()) 

def get_edit_function(from_word, to_word):
    for index,(c,h) in enumerate(zip(from_word,to_word)):
        if c != h:
            break
    else:
        index = max(len(from_word), len(to_word)) - 1
    if len(from_word) < len(to_word):
        return HashableInjector(function=lambda x,i : x[:i] + to_word[index] + x[i:],
                name="--->{}".format(to_word[index])), index
    if len(from_word) > len(to_word):
        return HashableInjector(function=lambda x,i : x[:i] + x[i+1:],
                name="{}--->".format(from_word[index])),index
    if index+1 >= len(from_word) or from_word[index+1] == to_word[index+1]:
        return HashableInjector(function=lambda x,i : x[:i] + to_word[index] + x[i+1:],
                name="{}-->{}".format(from_word[index], to_word[index])), index
    return HashableInjector(function=lambda x,i : x[:i] + x[i+1] + x[i] + x[i+2:],
            name="{}<->{}".format(from_word[index], from_word[index+1])),index

def error_iterator(fn, dictionary, whitelist=None):
    #TODO: utf-8 shouldn't be hardcoded
    if whitelist is None:
        whitelist = set()
    num_correct = 0
    num_incorrect = 0
    with codecs.open(fn, "r", "utf-8") as f:
        linecount = sum(1 for line in f)
    with codecs.open(fn, "r", "utf-8") as f:
        for line in tqdm(f, total=linecount):
            try:
                lang = detect(line)
            except Exception:
                continue
            if lang != "en":
                continue
            for word in tokenize(line.lower()):
                #some adhoc stemming
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
                try:
                    corrected = dictionary.correct(word).lower()
                except IndexError:
                    continue
                if corrected == word:
                    num_correct += 1
                elif corrected:
                    num_incorrect += 1
                    yield word, corrected

class ErrorInjector(object):

    def __init__(self, source_fn, dictionary, ngram_size=3, whitelist=None):
        self.generators = defaultdict(Counter)
        self.statistics = Counter()
        self.ngram_size = ngram_size
        self._train(source_fn, dictionary, whitelist)

    def _train(self, fn, dictionary, whitelist):
        self.global_error_count = 0
        for incorrect, correct in error_iterator(fn, dictionary, whitelist=whitelist):
            if dl_distance(incorrect, correct) == 1:
                self.global_error_count += 1
                edit_function, index = get_edit_function(incorrect, correct)
                padded = "-"*(self.ngram_size-1) + correct
                index += self.ngram_size-1
                trigram = padded[index-2:index+1]
                self.generators[trigram][edit_function] += 1

    def inject_errors(self, word):
        results = []
        for i,ngram in enumerate(ngram_generator(word, self.ngram_size)):
            gens = self.generators[ngram]
            for gen,weight in gens.iteritems():
                corrupted = gen.function(word, i)
                prob = weight/float(self.global_error_count)
                results.append(Error(correct=word, incorrect=corrupted, 
                                      error_name=gen.name, probability=prob))
        return results
