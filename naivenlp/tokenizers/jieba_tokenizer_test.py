import unittest

from .jieba_tokenizer import JiebaTokenizer


class JiebaTokenizerTest(unittest.TestCase):

    def testJiebaTokenizer(self):
        tokenizer = JiebaTokenizer(
            vocab_file=None,
            dict_files=[
                #    'naivenlp/tokenizers/data/dict.txt',
                'data/jieba/hello.txt'
            ])
        tokenizer.initialize()
        words = [w for w in tokenizer.tokenize('我在上海工作', cut_all=True, hmm=False)]
        print(words)

        words = [w for w in tokenizer.tokenize('我在上海工作', cut_all=False, hmm=False)]
        print(words)

        words = [w for w in tokenizer.tokenize('我在上海工作', cut_all=False, hmm=True)]
        print(words)

        print(list(tokenizer.words_freq.items())[:10])
        print(tokenizer.special_tokens)


if __name__ == "__main__":
    unittest.main()
