from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, StrictGlobalSequenceAligner

class TooManyAlignmentsError(Exception):
    def __init__(self, word1, word2, num_alignments):
        self.__dict__.update(locals())
        del self.self

    def __str__(self):
        return ("Too many alignments (%d) from '%s' to '%s' (max=1)" %
            (self.num_alignments, self.word1, self.word2))

class TooManyEditsError(Exception):
    def __init__(self, word1, word2, num_edits, max_edits):
        self.__dict__.update(locals())
        del self.self

    def __str__(self):
        return ("Too many edits (%d) from '%s' to '%s' (max=%d)" %
            (self.num_edits, self.word1, self.word2, self.max_edits))

class EditFinder(object):
    def __init__(self, max_edits=1):
        self.__dict__.update(locals())
        del self.self

    def find(self, error, word):
        v = Vocabulary()
        a = v.encodeSequence(Sequence(error))
        b = v.encodeSequence(Sequence(word))
    
        scoring = SimpleScoring(2, -1)
        aligner = StrictGlobalSequenceAligner(scoring, -2)
        score, encodeds = aligner.align(a, b, backtrace=True)
    
        if len(encodeds) > 1:
            raise TooManyAlignmentsError(error, word, len(encodeds))
    
        encoded = encodeds[0]
        alignment = v.decodeSequenceAlignment(encoded)
        edits = []
        for i in range(len(alignment.first)):
            if alignment.first[i] != alignment.second[i]:
                edits.append((alignment.first[i], alignment.second[i]))

        if len(edits) > self.max_edits:
            raise TooManyEditsError(error, word, len(edits), self.max_edits)

        return edits
