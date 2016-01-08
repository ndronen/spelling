import operator
import progressbar
from nltk.stem import PorterStemmer
from spelling.dictionary import Enchant
from spelling.utils import build_progressbar

class Job(object):
    def run(self, **kwargs):
        raise NotImplementedError()

class DistanceToNearestStem(Job):
    def run(self, words=None, distance=None, dictionary=Enchant()):
        """
        words : list
            The words for which to find 
        distance : callable
            Function taking two words and returning a distance.
        dictionary : spelling.dictionary 
            Instance of a class in spelling.dictionary.
        """
        nearest = []
        pbar = build_progressbar(words)
        for i,word in enumerate(words):
            pbar.update(i+1)
            suggestions = self.suggest(word, dictionary)
            if len(suggestions) == 0:
                nearest.append((word, "", 100))
            else:
                distances = [(word, s, distance(word, s))
                    for s in suggestions]
                sorted_distances = sorted(distances,
                    key=operator.itemgetter(2))
                nearest.append(sorted_distances[0])
        pbar.finish()
        return pd.DataFrame(data=nearest,
            columns=['word', 'suggestion', 'distance'])

    def suggest(self, word, dictionary, stemmer=PorterStemmer()):
        """
        Get suggested replacements for a given word from a dictionary.
        Remove words that have the same stem or are otherwise too similar.
        """
        stemmed_word = stemmer.stem(word)
        suggestions = []
        for suggestion in dictionary.suggest(word):
            if stemmer.stem(suggestion) == stemmed_word:
                continue
            # Will need to handle some cases here that Porter misses.
            suggestions.append(suggestion)
        return suggestions

