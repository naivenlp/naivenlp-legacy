import abc
import logging
import os


class AbstractTokenizer(abc.ABC):

    def tokenize(self, inputs, **kwargs):
        raise NotImplementedError()

    def encode(self, inputs, **kwargs):
        raise NotImplementedError()

    def decode(self, inputs, **kwargs):
        raise NotImplementedError()


class VocabBasedTokenizer(AbstractTokenizer):

    def __init__(self,
                 vocab_file,
                 pad_token='[PAD]',
                 unk_token='[UNK]',
                 bos_token='[BOS]',
                 eos_token='[EOS]',
                 **kwargs):
        self.vocab_file = vocab_file
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

        self.vocab = self._load_vocab(vocab_file)
        self.reverse_vocab = self._reverse_vocab()

    def _load_vocab(self, file):
        vocabs = dict(self.special_tokens)
        if not file:
            logging.warning('vocab_file is empty or None.')
            return vocabs
        if not os.path.exists(file):
            logging.warning('vocab file {} does not exists.'.format(file))
            return vocabs
        idx = len(vocabs)
        with open(file, mode='rt', encoding='utf8') as fin:
            for line in fin:
                word = line.rstrip('\n').strip()
                if not word:
                    continue
                if word in vocabs:
                    continue
                vocabs[word] = idx
                idx += 1
        return vocabs

    def _reverse_vocab(self):
        reverse_vocabs = {}
        for k, v in self.vocab.items():
            reverse_vocabs[v] = k
        return reverse_vocabs

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def special_tokens(self):
        return [(self.pad_token, self.pad_id), (self.unk_token, self.unk_id),
                (self.bos_token, self.bos_id), (self.eos_token, self.eos_id)]

    def tokenize(self, inputs, **kwargs):
        raise NotImplementedError()

    def encode(self, inputs, add_bos=False, add_eos=False, **kwargs):
        tokens = self.tokenize(inputs, **kwargs)
        ids = [self.vocab.get(t, self.unk_id) for t in tokens]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, inputs, drop_bos=True, drop_eos=True, **kwargs):
        tokens = [self.reverse_vocab.get(t, self.unk_token) for t in inputs]
        if not tokens:
            return []
        if drop_bos and tokens[0] == self.bos_token:
            tokens = tokens[1:]
        if drop_eos and tokens[-1] == self.eos_token:
            tokens = tokens[:-1]
        return tokens
