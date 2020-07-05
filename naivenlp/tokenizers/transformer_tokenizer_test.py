import unittest

from .transformer_tokenizer import TransformerTokenizer


class TransformerTokenizerTest(unittest.TestCase):

    def testTransformerTokenizer(self):
        tokenizer = TransformerTokenizer(
            vocab_file='testdata/vocab_chinese.txt',
            bos_token='<S>',
            eos_token='<T>',)
        self.assertEqual(0, tokenizer.pad_id)
        self.assertEqual(100, tokenizer.unk_id)
        self.assertEqual(104, tokenizer.bos_id)
        self.assertEqual(105, tokenizer.eos_id)


if __name__ == "__main__":
    unittest.main()
