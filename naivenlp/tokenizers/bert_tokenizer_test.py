import unittest

from .bert_tokenizer import BertTokenizer


class BertTokenizerTest(unittest.TestCase):

    def testBertTokenizer(self):
        tokenizer = BertTokenizer(vocab_file=None)
        self.assertEqual(0, tokenizer.pad_id)
        self.assertEqual(1, tokenizer.unk_id)
        self.assertEqual(2, tokenizer.bos_id)
        self.assertEqual(3, tokenizer.eos_id)
        self.assertEqual(4, tokenizer.cls_id)
        self.assertEqual(5, tokenizer.sep_id)
        self.assertEqual(6, tokenizer.mask_id)


if __name__ == "__main__":
    unittest.main()
