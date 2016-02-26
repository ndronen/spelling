import unittest
import pandas as pd
from spelling.baseline import CharacterLanguageModel

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

if __name__ == '__main__':
    unittest.main()
