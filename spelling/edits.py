import sys
from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, StrictGlobalSequenceAligner

def subsequences(word):
    """
    Returns a list of all subsequences of a word.
    """
    for i in range(0, len(word)):
        for j in range(i+1, len(word[i:])+1):
            yield word[i:j]

class EditFinder(object):
    def __init__(self):
        self.__dict__.update(locals())
        del self.self
        self.scoring = SimpleScoring(2, -1)
        self.aligner = StrictGlobalSequenceAligner(self.scoring, -2)

    def align(self, word, error):
        vocab = Vocabulary()
        a = vocab.encodeSequence(Sequence(word))
        b = vocab.encodeSequence(Sequence(error))
        score, encodings = self.aligner.align(a, b, backtrace=True)
    
        # Choose the highest-score alignment.
        score = -sys.maxsize
        best_alignment = None
        for encoding in encodings:
            alignment = vocab.decodeSequenceAlignment(encoding)
            if alignment.score > score:
                best_alignment = alignment
                score = alignment.score

        return best_alignment.first, best_alignment.second

    def edit_is_rotation(self, first, second, start, end):
        first_span = first[start:end+1]
        second_span = [c for c in reversed(second[start:end+1])]
        return first_span == second_span and \
            '-' not in first_span and '-' not in second_span

    def build_rotation(self, first, second, start, end):
        first_span = first[start:end+1]
        second_span = second[start:end+1]
        if start == 0:
            first_span.insert(0, '^')
            second_span.insert(0, '^')
        else:
            first_span.insert(0, first[start-1])
            second_span.insert(0, first[start-1])
        return (''.join(first_span), ''.join(second_span))

    def edit_is_transposition(self, first, second, start, end):
        first_span = first[start:end+1]
        second_span = [c for c in reversed(second[start:end+1])]
        return first_span == second_span and \
            first_span[0] == '-' and second_span[0] == '-'

    def build_transposition(self, first, second, start, end):
        first_span = first[start+1:end+1]
        second_span = second[start:end]
        return (''.join(first_span), ''.join(second_span))

    def edit_is_insertion(self, first, second, start, end):
        ret = first[start] == '-'
        #print('edit_is_insertion', first, second, start, end, first[start] == '-', ret)
        return ret

    def build_insertion(self, first, second, start, end):
        extent = 0
        for c in first[start:]:
            if c != '-':
                break
            extent += 1
        if start == 0:
            first_span = "^" + ''.join(first[:1])
            second_span = first_span[:-1] + ''.join(second[max(0,start-1):start+extent])
        else:
            first_span = ''.join(first[max(0,start-2):start+1])
            second_span = first_span[:-2] + ''.join(second[max(0,start-1):start+extent])

        first_span = ''.join(c for c in first_span if c != "-")
        second_span = ''.join(c for c in second_span if c != "-")
        return (first_span, second_span)

    def edit_is_deletion(self, first, second, start, end):
        #ret = start == end and second[start] == '-'
        ret = second[start] == '-'
        #print('edit_is_deletion', first, second, start, end, second[start] == '-', ret)
        return ret

    def build_deletion(self, first, second, start, end):
        #print('build_deletion', first, second, start, end, len(first))
        extent = 0
        for c in second[start:]:
            if c != '-':
                break
            extent += 1
        if start == 0:
            first_span = '^' + first[start]
            second_span = '^'
        else:
            first_span = ''.join(first[start-1:start+extent])
            second_span = first[start-1]
        return (first_span, second_span)

    def edit_is_substitution(self, first, second, start, end):
        ret = '-' not in [first[start], second[start]] and \
                first[start] != second[start]
        #print('edit_is_substitution', first, second, start, end, first[start] == '-', ret)
        return ret

    def build_substitution(self, first, second, start, end):
        #print('build_substitution', first, second, start, end)
        return (''.join(first[max(0,start-1):start+1]), ''.join(second[max(0,start-1):start+1]))
        #return (first[start], second[start])

    def build_edits(self, first, second):
        positions = []

        for i in range(len(first)):
            if first[i] != second[i]:
                positions.append(i)

        edits = []
        edit_indices = []

        #print('positions', positions)

        skip_next = 0

        for i in range(len(positions)):
            start = positions[i]
            try:
                end = positions[i+1]
            except IndexError:
                end = start

            if skip_next:
                skip_next -= positions[i] - positions[i-1]
                #print skip_next
            if skip_next:
                if skip_next > 0:
                    continue
                else:
                    skip_next = 0

            #print('i', i, 'start', start, 'end', end, 'edits', edits)

            edit_indices.append(i)

            if self.edit_is_rotation(first, second, start, end):
                #print('found a rotation in ' + str(first) + ' -> ' + str(second))
                edits.append(self.build_rotation(first, second, start, end))
                skip_next = len(edits[-1][1])
            elif self.edit_is_transposition(first, second, start, end):
                #print('found a transposition in ' + str(first) + ' -> ' + str(second))
                edits.append(self.build_transposition(first, second, start, end))
                skip_next = 3
            elif self.edit_is_insertion(first, second, start, end):
                #print('found an insertion in ' + str(first) + ' -> ' + str(second))
                edits.append(self.build_insertion(first, second, start, end))
                skip_next = len(edits[-1][1])-1
                #print "for edit",edits[-1]
                #print "setting skip next to",skip_next
            elif self.edit_is_deletion(first, second, start, end):
                #print('found a deletion in ' + str(first) + ' -> ' + str(second))
                edits.append(self.build_deletion(first, second, start, end))
                skip_next = len(edits[-1][0])-1
            elif self.edit_is_substitution(first, second, start, end):
                #print('found a substitution in ' + str(first) + ' -> ' + str(second))
                edits.append(self.build_substitution(first, second, start, end))
            else:
                raise ValueError('did not find any edits in %s => %s' % (
                    first, second))

        return edits

    def find(self, word, error):
        first, second = self.align(word, error)
        edits = self.build_edits(first, second)
        return edits

    def apply(self, word, edits):
        word = "^" + word
        planned = []
        for from_gram, to_gram in edits:
            index = word.find(from_gram)
            if index != -1:
                planned.append((index, len(from_gram), len(to_gram), to_gram))
        if len(planned) < len(edits):
            raise ValueError('could not apply all edits to "%s"' % word)
        planned.sort(reverse=True)
        new_word = word
        for index, size, _, to_gram in planned:
            #print new_word
            new_word = new_word[:index] + to_gram + new_word[index+size:]
        new_word = new_word.strip("^")
        return new_word

    def remove_dashes(self, index, word):
        new_word = []
        new_index = index
        for i,c in enumerate(word):
            if c == "-":
                if i < index:
                    new_index -= 1
            else:
                new_word.append(c)
        return new_index, ''.join(new_word)

class Editor(object):
    def __init__(self):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'

    def edits(self, word):
        return set(self.split(word) +
                self.delete(word) +
                self.transpose(word) + 
                self.substitute(word) + 
                self.insert(word))

    def edit(self, word, operation):
        f = getattr(self, operation)
        return f(word)
                                                            
    def split(self, word):
        split = []
        for i in range(len(word) + 1):
            split.append((word[:i] + ' ' + word[i:]))
        return split

    def delete(self, word):
        delete = []
        for i in range(len(word)):
            delete.append(word[:i] + word[i+1:])
        return delete

    def transpose(self, word):
        transpose = []
        for i in range(len(word)-1):
            transpose.append(word[:i] + word[i+1] + word[i] + word[i+2:])
        return transpose

    def substitute(self, word):
        substitute = []
        for c in self.alphabet:
            for i in range(0, len(word)):
                substitute.append(word[:i] + c + word[i+1:])
        return substitute

    def insert(self, word):
        insert = []
        for c in self.alphabet:
            for i in range(0, len(word)):
                insert.append(word[:i] + c + word[i:])
        return insert

class EditConstraintError(Exception):
    pass

class EditConstraints(object):
    """
    Return true if the edit changes the first character, unless
    the edit is one that we explicitly allow.
    """
    @staticmethod
    def edit_changes_first_character(word, edit):
        result = word.startswith(edit[0][0]) and \
                not word.startswith(edit[1][0])
        return result
