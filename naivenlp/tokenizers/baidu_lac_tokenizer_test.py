import unittest

from LAC import LAC

from .baidu_lac_tokenizer import BaiduLACTokenizer


class BaiduLACTokenizerTest(unittest.TestCase):

    def testTokenization(self):
        tokenizer = BaiduLACTokenizer(vocab_file=None)
        lac = LAC(mode='seg')

        sentences = [
            '我在上海工作',
            '我来到北京清华大学',
            '乒乓球拍卖完了',
            '中国科学技术大学',
        ]
        for sent in sentences:
            self.assertEqual(lac.run(sent), tokenizer.tokenize(sent))


if __name__ == "__main__":
    unittest.main()
