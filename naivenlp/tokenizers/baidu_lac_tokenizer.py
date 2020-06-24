import logging

from LAC import LAC

from .abstract_tokenizer import VocabBasedTokenizer


class BaiduLACTokenizer(VocabBasedTokenizer):

    def __init__(self,
                 vocab_file,
                 custom_dict_file=None,
                 pad_token='[PAD]',
                 unk_token='[UNK]',
                 bos_token='[BOS]',
                 eos_token='[EOS]',
                 **kwargs):
        super().__init__(
            vocab_file, pad_token=pad_token, unk_token=unk_token, bos_token=bos_token, eos_token=eos_token, **kwargs)
        self.lac = LAC(mode='seg')
        if custom_dict_file:
            self.lac.load_customization(custom_dict_file)
            logging.info('Load custom dict: {} successfully fro LAC.'.format(custom_dict_file))

    def tokenize(self, inputs, **kwargs):
        return self.lac.run(inputs)
