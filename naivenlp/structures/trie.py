import abc


class Node:

    def __init__(self):
        self.val = None
        self.children = {}


class AbstractTrie(abc.ABC):
    """Trie interface."""

    def put(self, sequence, **kwargs):
        """Insert an sequence"""
        raise NotImplementedError()

    def get(self, key, **kwargs):
        """Get node by key sequence"""
        raise NotImplementedError()

    def delete(self, key, **kwargs):
        """Delete an sequence"""
        raise NotImplementedError()

    def contains(self, key, **kwargs):
        """Contains an sequence or not"""
        raise NotImplementedError()

    def is_empty(self, **kwargs):
        """The trie is empty or not"""
        raise NotImplementedError()

    def longest_prefix_of(self, sequence, **kwargs):
        """Longest prefix of the sequence in the trie"""
        raise NotImplementedError()

    def keys_with_prefix(self, prefix, **kwargs):
        """Collect all of the sequence in the trie that has this prefix"""
        raise NotImplementedError()

    def size(self, **kwargs):
        """Number of leaf nodes of this trie"""
        raise NotImplementedError()

    def show(self, **kwargs):
        """Print this trie"""
        raise NotImplementedError()


class Trie(AbstractTrie):

    def __init__(self):
        self.root = Node()
        self.root.val = '.'

    def _put(self, node, sequence, depth):
        if depth == len(sequence):
            return
        v = sequence[depth]
        if v not in node.children:
            n = Node()
            n.val = v
            node.children[v] = n
        n = node.children[v]
        return self._put(n, sequence, depth + 1)

    def put(self, sequence, **kwargs):
        self._put(self.root, sequence, 0)

    def _get(self, node, sequence, depth):
        if depth == len(sequence):
            # return sequence
            return node
        if not sequence[depth] in node.children:
            return None
        n = node.children[sequence[depth]]
        return self._get(n, sequence, depth + 1)

    def get(self, sequence, **kwargs):
        return self._get(self.root, sequence, 0)

    def _delete(self, node, sequence, depth):
        if depth == len(sequence) - 1:
            node.children.pop(sequence[depth], None)
            return
        self._delete(node.children[sequence[depth]], sequence, depth + 1)

    def delete(self, sequence, **kwargs):
        return self._delete(self.root, sequence, 0)

    def contains(self, key, **kwargs):
        node = self.get(key)
        return True if node is not None else False

    def is_empty(self, **kwargs):
        return len(self.root.children) == 0

    def _search(self, node, sequence, depth):
        if not node.children:
            return sequence[:depth]
        if depth == len(sequence):
            return sequence

        if sequence[depth] not in node.children:
            return sequence[:depth]
        return self._search(node.children[sequence[depth]], sequence, depth + 1)

    def longest_prefix_of(self, sequence, **kwargs):
        return self._search(self.root, sequence, 0)

    def _collect(self, node, prefix):
        prefix = list(prefix)
        results = []

        def _traversal(n, arr):
            if not n.children:
                results.append(prefix + arr)
                return
            for k, c in n.children.items():
                arr.append(k)
                _traversal(c, arr[:])
                arr.pop()

        _traversal(node, [])
        return results

    def keys_with_prefix(self, prefix, **kwargs):
        node = self.get(prefix)
        if node is None:
            return []
        return self._collect(node, prefix)

    def size(self, **kwargs):

        def _count(node):
            if not node.children:
                return 1
            c = 0
            for k, n in node.children.items():
                c += _count(n)
            return c

        return _count(self.root)

    def _show(self, node, depth, max_depth=None):
        if not node:
            return
        if max_depth is not None and depth == max_depth + 1:
            return
        if depth == 0:
            print('.')
        for k, n in node.children.items():
            print("|    " * depth + '+----' + n.val)
            self._show(n, depth + 1, max_depth)

    def show(self, max_depth=None):
        self._show(self.root, 0, max_depth=max_depth)
