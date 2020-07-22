import unittest


from .deep_corrector import TransformerCorrector


class DeepCorrectorTest(unittest.TestCase):

    def testTransformerCorrector(self):
        c = TransformerCorrector(saved_model='models/correction_models/transformer-step-2000')
        inputs = [
            '我最近每天晚上都会拧着鼻子去喝30cc的醋了。',
            '我拍了一张樱花的照片。',
            '它是星星形状的娇小的糖果，带着昔日的风貌。',
            '羡慕啊。',
            '你重视旋律还是重视歌词？',
            '今天我去看了电影。',
            '即使那样的一天，他们也不打伞，跑回家。',
            '我走啦！',
            '吃寿司卷的时候，不要一言不发。',
            '这个网点Lang8的朋友告诉我的。',
        ]

        for inp in inputs:
            print('inpiut: ', inp)
            res, prob = c.correct(inp)
            print('result: ', res)
            print('  prob: ', prob)
            print()


if __name__ == "__main__":
    unittest.main()
