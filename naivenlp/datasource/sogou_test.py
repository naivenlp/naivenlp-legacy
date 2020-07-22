import unittest

from . import sogou


class SogouTest(unittest.TestCase):

    def testDownload(self):
        sogou.download_category(1, '/tmp/')


if __name__ == "__main__":
    unittest.main()
