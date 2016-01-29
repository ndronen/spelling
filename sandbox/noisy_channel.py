from spelling.edit import EditFinder, TooManyEditsError

class NoisyChannelWithUnigramLanguageModel(object):
    """
    An implementation of the noisy channel model.  Given a misspelling
    x and a possible correction w, the model computes the probability
    that w is the correct word P(w|x) = P(x|w) P(w).  P(w) is the prior
    probability of w; here that prior probability comes from a unigram
    language model.  The channel, P(x|w), gives the probability of
    spelling w as x.

    channel : dict of dict
        A dictionary d such that d[A][B] is the probability of typing
        B instead of A.
    language_model : dict 
        A dictionary d such that d[W] is the prior probability of a word W.
    max_edits : int
        The maximum number of edits allowed by the channel.  Larger numbers
        of edits require the channel to contain more keys.
    """
    def __init__(self, channel, language_model, max_edits=1):
        self.__dict__.update(locals())
        self.finder = EditFinder(max_edits=max_edits)
        del self.self

    def deletion_prob(self, x, y):
        # How often xy is typed as y.
        raise NotImplementedError()
        pass

    def insertion_prob(self, x, y):
        # How often x is typed as xy.
        raise NotImplementedError()
        pass

    def substitution_prob(self, x, y):
        # How often x is replaced by y.
        raise NotImplementedError()
        pass

    def transposition_prob(self, x, y):
        # How often x and y are transposed.
        raise NotImplementedError()
        pass

    def _prior(self, word):
        return self.language_model[word]

    def _edits(self, error, word):
        return self.finder.find(error, word)

    def _conditional(self, error, word):
        # Get edit necessary to convert error to word.  (Edit is singular,
        # since we will start with distance 1 edits.)
        edits = self.edits(error, word)

    def probability(self, error, word):
        raise NotImplementedError()

        return np.log(self.conditional(error, word)) +
            np.log(self.prior(word))
