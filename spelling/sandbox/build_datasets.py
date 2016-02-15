raise ValueError('make this a spelling.job')

from spelling.mitton import CORPORA, build_mitton_datasets
import cPickle
import spelling.dictionary

constructors = [
        spelling.dictionary.Aspell,
        spelling.dictionary.AspellWithNorvigLanguageModel,
        spelling.dictionary.NorvigWithAspellVocabWithoutLanguageModel,
        spelling.dictionary.NorvigWithAspellVocabAndGoogleLanguageModel,
        spelling.dictionary.NorvigWithAspellVocabGoogleLanguageModelPhoneticCandidates
        ]

corpus_names = [c.replace('data/', '') for c in CORPORA]
corpus_names = [c.replace('.dat', '') for c in corpus_names]

for i, corpus in enumerate(CORPORA):
    corpus_name = corpus_names[i]
    print(corpus_name)
    dfs = build_mitton_datasets(corpus, constructors=constructors)
    cPickle.dump(dfs, open('data/' + corpus_name + '.pkl', 'w'))
