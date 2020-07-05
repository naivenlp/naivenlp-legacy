import unittest

from .bert_tokenizer import BertTokenizer


class BertTokenizerTest(unittest.TestCase):

    def testBertTokenizer(self):
        tokenizer = BertTokenizer(
            vocab_file='testdata/vocab_chinese.txt',
            bos_token='<S>',
            eos_token='<T>')
        for k, v in tokenizer.special_tokens():
            print(k, v)
            n = k.replace('_token', '_id')
            print(n, getattr(tokenizer, n))

        self.assertEqual(0, tokenizer.pad_id)
        self.assertEqual(100, tokenizer.unk_id)
        self.assertEqual(104, tokenizer.bos_id)
        self.assertEqual(105, tokenizer.eos_id)
        self.assertEqual(101, tokenizer.cls_id)
        self.assertEqual(102, tokenizer.sep_id)
        self.assertEqual(103, tokenizer.mask_id)

        print(tokenizer.special_tokens())


if __name__ == "__main__":
    unittest.main()
