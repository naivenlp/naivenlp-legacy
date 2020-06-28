import unittest


from .detector import KenLMDetector


class DetectorTest(unittest.TestCase):

    def testKenLMDetector(self):
        detector = KenLMDetector(kenlm_model_path=None)
        outputs = detector.detect('andorid工程师')
        print('outputs size: ', len(outputs))
        for o in outputs:
            print(o)


if __name__ == "__main__":
    unittest.main()
