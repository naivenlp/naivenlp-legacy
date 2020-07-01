import unittest

from .corrector import KenLMCorrector


class CorrectorTest(unittest.TestCase):

    def testKenLMCorrector(self):
        c = KenLMCorrector()
        self.assertEqual(1396, len(c.stop_words))
        self.assertEqual(631928, len(c.word_freq_map))
        self.assertEqual(1366, len(c.confusion_map))
        self.assertEqual(3401, len(c.same_pinyin_chars))
        self.assertEqual(345, len(c.same_stroke_chars))

        texts = [
            '软件开发工成师',
            '少先队员因该为老人让坐',
        ]

        for text in texts:
            res = c.correct(text, threshold=0.5)
            print(' original text: {}'.format(text))
            print('corrected text: {}'.format(res))


if __name__ == "__main__":
    unittest.main()
