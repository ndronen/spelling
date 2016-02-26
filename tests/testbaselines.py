import unittest
import numpy as np
import pandas as pd
from spelling.baseline import CharacterLanguageModel, LanguageModelClassifier, LanguageModelClassifier, LanguageModelClassifier, LanguageModelClassifier

class TestCharacterLanguageModel(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv('data/aspell-dict.csv.gz', sep='\t', encoding='utf8')
        self.words = df.word.tolist()

    def test_train_test_are_files(self):
        lm = CharacterLanguageModel('witten-bell', order=3)
        lm.fit(self.words)
        output = lm.predict(self.words)
        log_probs = output['log_probs']
        ppls = output['ppls']
        ppl1s = output['ppl1s']

        self.assertEquals(len(self.words), len(log_probs))
        self.assertEquals(len(self.words), len(ppls))
        self.assertEquals(len(self.words), len(ppl1s))

class TestLanguageModelClassifier(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv('data/aspell-dict.csv.gz', sep='\t', encoding='utf8')
        self.words = df.word.tolist()

    def test_language_model_classifier(self):
        lm_real_words = CharacterLanguageModel('witten-bell', order=3)
        lm_real_words.fit(self.words)

        real_words = self.words
        non_words = lm_real_words.generate(1, len(real_words))

        lm_non_words = CharacterLanguageModel('witten-bell', order=3)
        lm_non_words.fit(non_words)

        clf = LanguageModelClassifier([lm_non_words, lm_real_words])
        real_words_pred = clf.predict(real_words)
        non_words_pred = clf.predict(non_words)

        real_words_bincount = np.bincount(real_words_pred)
        non_words_bincount = np.bincount(non_words_pred)

        self.assertTrue(real_words_bincount[0] < real_words_bincount[1])
        self.assertTrue(non_words_bincount[0] > non_words_bincount[1])

if __name__ == '__main__':
    unittest.main()
