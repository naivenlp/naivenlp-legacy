import abc
import logging
import os


class AbstractTokenizer(abc.ABC):

    def tokenize(self, inputs, **kwargs):
        raise NotImplementedError()

    def tokens2ids(self, inputs, **kwargs):
        raise NotImplementedError()

    def ids2tokens(self, inputs, **kwargs):
        raise NotImplementedError()

    def encode(self, inputs, **kwargs):
        raise NotImplementedError()

    def decode(self, inputs, **kwargs):
        raise NotImplementedError()


class VocabBasedTokenizer(AbstractTokenizer):

    def __init__(self, vocab_file, **kwargs):
        self.vocab_file = vocab_file

        self._special_tokens = []
        for k, v in kwargs.items():
            if str(k).endswith('_token') and v is not None:
                self._special_tokens.append((k, v))

        self.vocab = self._build_vocab(vocab_file)
        self.reverse_vocab = self._reverse_vocab()

        for k, v in self._special_tokens:
            _id = str(k.split('_')[0]) + '_id'
            setattr(self, k, v)
            setattr(self, _id, self.vocab[v])

    def _build_vocab(self, file):
        if not file:
            logging.warning('vocab_file is empty or None.')
            return {}
        words = []
        with open(file, mode='rt', encoding='utf8') as fin:
            for line in fin:
                line = line.strip('\n').strip()
                if not line:
                    continue
                word = line.strip()
                words.append(word)

        vocab = set(words)
        special_tokens = [v for _, v in self._special_tokens]
        for t in special_tokens:
            if t not in vocab:
                words.append(t)

        d = {}
        for i in range(len(words)):
            d[words[i]] = i
        return d

    def _reverse_vocab(self):
        reverse_vocabs = {}
        for k, v in self.vocab.items():
            reverse_vocabs[v] = k
        return reverse_vocabs

    @property
    def vocab_size(self):
        return len(self.vocab)

    def special_tokens(self):
        return self._special_tokens

    def tokenize(self, inputs, **kwargs):
        raise NotImplementedError()

    def tokens2ids(self, tokens, add_bos=False, add_eos=False, **kwargs):
        ids = [self.vocab.get(t, self.unk_id) for t in tokens]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def ids2tokens(self, ids, drop_bos=False, drop_eos=False, **kwargs):
        tokens = [self.reverse_vocab.get(t, self.unk_token) for t in ids]
        if drop_bos and tokens[0] == self.bos_token:
            tokens = tokens[1:]
        if drop_eos and tokens[-1] == self.eos_token:
            tokens = tokens[:-1]
        return tokens

    def encode(self, inputs, add_bos=False, add_eos=False, **kwargs):
        tokens = self.tokenize(inputs, **kwargs)
        return self.tokens2ids(tokens, add_bos=add_bos, add_eos=add_eos, **kwargs)

    def decode(self, inputs, drop_bos=True, drop_eos=True, **kwargs):
        ids = [self.reverse_vocab.get(t, self.unk_token) for t in inputs]
        if not ids:
            return []
        return self.ids2tokens(ids, drop_bos=drop_bos, drop_eos=drop_eos, **kwargs)


class CustomTokenizer(VocabBasedTokenizer):

    def __init__(self,
                 vocab_file,
                 tokenize_fn,
                 pad_token='[PAD]',
                 unk_token='[UNK]',
                 bos_token='[BOS]',
                 eos_token='[EOS]',
                 **kwargs):
        super().__init__(
            vocab_file,
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs)

        self.tokenize_fn = tokenize_fn

    def tokenize(self, inputs, **kwargs):
        return self.tokenize_fn(inputs, **kwargs)
