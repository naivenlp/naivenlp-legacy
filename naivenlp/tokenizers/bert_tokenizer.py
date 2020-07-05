from .abstract_tokenizer import VocabBasedTokenizer
from .tokenizer import BasicTokenizer, WordpieceTokenizer
from .transformer_tokenizer import TransformerTokenizer


class BertTokenizer(TransformerTokenizer):

    def __init__(self,
                 vocab_file,
                 pad_token='[PAD]',
                 unk_token='[UNK]',
                 bos_token='[BOS]',
                 eos_token='[EOS]',
                 cls_token='[CLS]',
                 sep_token='[SEP]',
                 mask_token='[MASK]',
                 do_lower_case=True,
                 do_basic_tokenization=True,
                 tokenize_chinese_chars=True,
                 never_split=None,
                 max_input_chars_per_word=100,
                 **kwargs):

        super().__init__(
            vocab_file,
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            cls_token=cls_token,
            sep_token=sep_token,
            mask_token=mask_token,
            do_lower_case=True,
            do_basic_tokenization=True,
            tokenize_chinese_chars=True,
            never_split=None,
            max_input_chars_per_word=100,
            **kwargs)
