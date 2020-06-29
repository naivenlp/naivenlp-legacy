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
        text = (c.correct('工成师'))
        print('text: {}'.format(text))


if __name__ == "__main__":
    unittest.main()
