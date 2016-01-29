import spelling.mitton

def count(corpus_file):
    words = spelling.mitton.load_mitton_words(corpus_file)
    pairs = spelling.mitton.build_mitton_pairs(words)
    error_count = len(pairs)
    word_count = len(set([p[1] for p in pairs]))

    corpus = corpus_file.replace('data/', '')
    corpus = corpus.replace('.dat', '')
    corpus = corpus.replace('-missp', '')
    corpus = corpus.title()

    return corpus, word_count, error_count

counts = []
for corpus in spelling.mitton.CORPORA:
    counts.append(count(corpus))
df = pd.DataFrame(data=counts, columns=['Corpus', 'Words', 'Errors'])
df = pd.concat([df, pd.DataFrame({
        'Corpus': ['Total'],
        'Words': [df.Words.sum()],
        'Errors': [df.Errors.sum()] 
        })])
df.to_latex()
