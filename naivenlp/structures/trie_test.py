import unittest

from .trie import Trie


class TrieTest(unittest.TestCase):

    def testTrie(self):
        t = Trie()
        t.put("上海市 浦东新区".split())
        t.put("上海市浦东新区")
        print()
        t.show()

        self.assertEqual('市', t.get('上海市').val)
        self.assertEqual('浦东新区', t.get('上海市 浦东新区'.split()).val)
        self.assertEqual('市', t.get('上海市').val)
        self.assertEqual('区', t.get('上海市浦东新区').val)
        self.assertEqual(None, t.get('上海市浦东区'))
        self.assertEqual(None, t.get('上海市浦东新区哈哈哈'))

        self.assertEqual(2, t.size())

        self.assertEqual(True, t.contains('上海市 浦东新区'.split()))
        self.assertEqual(True, t.contains('上海市浦东新区'))
        self.assertEqual(True, t.contains('上海市浦东'))

        self.assertEqual('上海市浦东新区', t.longest_prefix_of('上海市浦东新区'))
        self.assertEqual('上海市浦东新', t.longest_prefix_of('上海市浦东新'))
        self.assertEqual(['上海市', '浦东新区'], t.longest_prefix_of('上海市 浦东新区'.split()))

        t.put('上海市黄浦区')
        for r in t.keys_with_prefix('上海市'):
            print(r)
            print('=' * 80)
        self.assertEqual(3, t.size())

        t.delete('上海市浦东')
        t.show()

        self.assertEqual(False, t.is_empty())

        print('=' * 80)

        t.delete('上')
        t.show()

        t.delete('上海市'.split())
        t.show()
        self.assertEqual(True, t.is_empty())


if __name__ == "__main__":
    unittest.main()
