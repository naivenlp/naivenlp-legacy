import logging
import os

import jieba

from .abstract_tokenizer import AbstractTokenizer, VocabBasedTokenizer

ACCURATE_MODE = 0
FULL_MODE = 1
SEARCH_MODE = 2


class JiebaTokenizer(VocabBasedTokenizer):

    def __init__(self,
                 vocab_file,
                 userdict_files=None,
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
        jieba.initialize()
        if userdict_files:
            for f in userdict_files:
                if not os.path.exists(f):
                    logging.warning('Load userdict: {} failed. File does not exist. Skipped.'.format(f))
                    continue
                with open(f, mode='rt', encoding='utf8') as fin:
                    jieba.load_userdict(fin)
                    logging.info('Load userdict: {} finished.'.format(f))

    def tokenize(self, inputs, mode=ACCURATE_MODE, hmm=True, **kwargs):
        if mode == ACCURATE_MODE:
            return [t for t in jieba.cut(inputs, cut_all=False, HMM=hmm)]
        elif mode == FULL_MODE:
            return [t for t in jieba.cut(inputs, cut_all=True, HMM=hmm)]
        elif mode == SEARCH_MODE:
            return [t for t in jieba.cut_for_search(inputs, HMM=hmm)]
        else:
            raise ValueError('Invalid mode: {}'.format(mode))
