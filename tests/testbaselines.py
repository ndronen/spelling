import unittest
import pandas as pd
from spelling.baseline import CharacterLanguageModel

class TestCharacterLanguageModel(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv('data/aspell-dict.csv.gz', sep='\t', encoding='utf8')
        self.words = df.word.tolist()

    def test_train_test_are_files(self):
        lm = CharacterLanguageModel(3)
        lm.fit(self.words)
        output = lm.predict(self.words)
        log_probs = lm.get_log_probs(output)
        ppls = lm.get_ppls(output)
        ppl1s = lm.get_ppl1s(output)

        self.assertEquals(len(self.words), len(log_probs))
        self.assertEquals(len(self.words), len(ppls))
        self.assertEquals(len(self.words), len(ppl1s))

if __name__ == '__main__':
    unittest.main()
