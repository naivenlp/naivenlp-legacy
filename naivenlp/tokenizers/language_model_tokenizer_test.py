import unittest

from .language_model_tokenizer import LanguageModelTokenizer, TransformerTokenizer


class LanguageModelTokenizerTest(unittest.TestCase):

    def testLanguageModelTokenizer(self):
        tokenizer = LanguageModelTokenizer(vocab_file=None)
        self.assertEqual(0, tokenizer.pad_id)
        self.assertEqual(1, tokenizer.unk_id)
        self.assertEqual(2, tokenizer.bos_id)
        self.assertEqual(3, tokenizer.eos_id)

    def testTransformerTokenizer(self):
        tokenizer = TransformerTokenizer(vocab_file=None)
        self.assertEqual(0, tokenizer.pad_id)
        self.assertEqual(1, tokenizer.unk_id)
        self.assertEqual(2, tokenizer.bos_id)
        self.assertEqual(3, tokenizer.eos_id)


if __name__ == "__main__":
    unittest.main()
