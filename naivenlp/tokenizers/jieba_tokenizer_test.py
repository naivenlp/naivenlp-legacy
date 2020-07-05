import unittest

import jieba

from .jieba_tokenizer import ACCURATE_MODE, FULL_MODE, SEARCH_MODE, JiebaTokenizer


class JiebaTokenizerTest(unittest.TestCase):

    def testTokenize(self):
        tokenizer = JiebaTokenizer(
            vocab_file='testdata/vocab_chinese.txt',
            userdict_files=[
                'data/jieba/hello.txt',
                'naivenlp/tokenizers/data/dict.txt',
            ])

        sentences = [
            '我在上海工作',
            '我来到北京清华大学',
            '乒乓球拍卖完了',
            '中国科学技术大学',
        ]

        for sent in sentences:
            self.assertEqual(
                [t for t in jieba.cut(sent, cut_all=False, HMM=True)],
                tokenizer.tokenize(sent, mode=ACCURATE_MODE, hmm=True))
            self.assertEqual(
                [t for t in jieba.cut(sent, cut_all=False, HMM=False)],
                tokenizer.tokenize(sent, mode=ACCURATE_MODE, hmm=False))
            self.assertEqual(
                [t for t in jieba.cut(sent, cut_all=True, HMM=True)],
                tokenizer.tokenize(sent, mode=FULL_MODE, hmm=True))
            self.assertEqual(
                [t for t in jieba.cut(sent, cut_all=True, HMM=False)],
                tokenizer.tokenize(sent, mode=FULL_MODE, hmm=True))
            self.assertEqual(
                [t for t in jieba.cut_for_search(sent, HMM=True)],
                tokenizer.tokenize(sent, mode=SEARCH_MODE, hmm=True))
            self.assertEqual(
                [t for t in jieba.cut_for_search(sent, HMM=False)],
                tokenizer.tokenize(sent, mode=SEARCH_MODE, hmm=False))


if __name__ == "__main__":
    unittest.main()
