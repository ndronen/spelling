"""
# coding: utf-8
non_word_non_word = get_ipython().magic(u'sx cat transpose-distance-1-non-words-lm-non-words.logprob')
len(non_word_non_word)
get_ipython().system(u'wc -l transpose-distance-1-non-words-lm-non-words.logprob')
non_word_real_word = get_ipython().magic(u'sx cat transpose-distance-1-non-words-lm-real-words.logprob')
real_word_non_word = get_ipython().magic(u'sx cat transpose-distance-1-real-words-lm-non-words.logprob')
real_word_real_word = get_ipython().magic(u'sx cat transpose-distance-1-real-words-lm-real-words.logprob')
len(real_word_non_word)
len(non_word_non_word)
len(real_word_real_word)
len(non_word_real_word)
real_word_non_word = np.array(real_word_non_word)
real_word_real_word = np.array(real_word_real_word)
none_word_non_word = np.array(non_word_non_word)
non_word_non_word = np.array(non_word_non_word)
del none_word_non_word
non_word_real_word = np.array(non_word_real_word)
#real_word_pred = 
real_word_pred = np.zeros_like(real_word_real_word)
non_word_pred = np.zeros_like(non_word_non_word)
real_word_pred[real_word_real_word > non_word_real_word] = 1
non_word_pred[real_word_non_word > non_word_non_word] = 1
np.bincount(non_word_pred)
non_word_pred
non_word_pred = np.zeros_like(non_word_non_word, dtype=np.int)
real_word_pred = np.zeros_like(real_word_real_word, dtype=np.int)
non_word_pred[real_word_non_word > non_word_non_word] = 1
real_word_pred[real_word_real_word > non_word_real_word] = 1
non_word_pred
np.bincount(non_word_pred)
np.bincount(real_word_pred)
real_word_real_word[0:10]
non_word_real_word[0:10]
target = [0 for r in real_word_non_word]
target.extend([1 for r in real_word_real_word])
#df = pd.DataFrame(
real_word = real_word_real_word.tolist()
real_word.extend(real_word_non_word.tolist())
non_word = non_word_real_word.tolist()
#non_word.extend(non_word_non_word.tolist())
#target = [0 for r in real_word_non_word]
real_word = real_word_non_word.tolist()
real_word.extend(real_word_real_word.tolist())
non_word = non_word_non_word.tolist()
non_word.extend(non_word_real_word.tolist())
target = [0 for r in real_word_non_word]
target.extend([1 for r in real_word_real_word])
df = pd.DataFrame({ 'target': target, 'non_word': non_word, 'real_word': real_word })
df.head()
df.tail()
np.exp(-11.5564)
np.exp(-11.5564) < np.exp(-16.3458)
#df.to_csv('
get_ipython().magic(u'ls ')
#df.to_csv('
#real_word_real_word = %sx cat transpose-distance-1-real-words-lm-real-words.logprob
df.to_csv('transpose-distance-1-lm-results.csv', index=False)
get_ipython().magic(u'history ')
get_ipython().magic(u'history ')
get_ipython().magic(u'save lm.py 1-59')
"""
