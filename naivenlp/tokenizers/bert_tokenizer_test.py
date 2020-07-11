import unittest

from .bert_tokenizer import BertTokenizer


class BertTokenizerTest(unittest.TestCase):

    def testBertTokenizer(self):
        tokenizer = BertTokenizer(
            vocab_file='testdata/vocab_chinese.txt',
            bos_token='<S>',
            eos_token='<T>')

        self.assertEqual(0, tokenizer.pad_id)
        self.assertEqual(100, tokenizer.unk_id)
        self.assertEqual(104, tokenizer.bos_id)
        self.assertEqual(105, tokenizer.eos_id)
        self.assertEqual(101, tokenizer.cls_id)
        self.assertEqual(102, tokenizer.sep_id)
        self.assertEqual(103, tokenizer.mask_id)

        tokenizer = BertTokenizer(vocab_file='testdata/vocab_chinese.txt', bos_token='<S>', eos_token='</S>')
        self.assertEqual(0, tokenizer.pad_id)
        self.assertEqual(100, tokenizer.unk_id)
        self.assertEqual(104, tokenizer.bos_id)
        self.assertEqual(21127, tokenizer.eos_id)
        self.assertEqual(101, tokenizer.cls_id)
        self.assertEqual(102, tokenizer.sep_id)
        self.assertEqual(103, tokenizer.mask_id)

        tokenizer = BertTokenizer(
            vocab_file='testdata/vocab_chinese.txt', bos_token='<S>', eos_token='</S>', xxx_token='XXX')
        self.assertEqual(0, tokenizer.pad_id)
        self.assertEqual(100, tokenizer.unk_id)
        self.assertEqual(104, tokenizer.bos_id)
        self.assertEqual(21127, tokenizer.eos_id)
        self.assertEqual(101, tokenizer.cls_id)
        self.assertEqual(102, tokenizer.sep_id)
        self.assertEqual(103, tokenizer.mask_id)
        self.assertEqual(21128, tokenizer.xxx_id)


if __name__ == "__main__":
    unittest.main()
