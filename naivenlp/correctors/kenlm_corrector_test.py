import unittest

from .kenlm_corrector import KenLMCorrector


class KenLMCorrectorTest(unittest.TestCase):

    def testKenLMCorrector(self):
        c = KenLMCorrector()

        print('chinese char set size: {}'.format(len(c.chinese_char_set)))
        print('same pinyin map size: {}'.format(len(c.same_pinyin_map)))
        print('same stroke map size: {}'.format(len(c.same_stroke_map)))

        texts = [
            '软件开发工成师',
            '少先队员因该为老人让坐',
        ]

        for text in texts:
            print(c.correct(text))
            self.assertEqual(c.correct(text)[0], c.corrector.correct(text)[0])


if __name__ == "__main__":
    unittest.main()
