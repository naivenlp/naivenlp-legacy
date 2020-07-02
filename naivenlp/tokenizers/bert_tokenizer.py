from .language_model_tokenizer import LanguageModelTokenizer
from .tokenizer import BasicTokenizer, WordpieceTokenizer


class BertTokenizer(LanguageModelTokenizer):

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

        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token
        self.cls_id = 4
        self.sep_id = 5
        self.mask_id = 6

        super().__init__(
            vocab_file,
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            do_lower_case=do_lower_case,
            do_basic_tokenization=do_basic_tokenization,
            tokenize_chinese_chars=tokenize_chinese_chars,
            never_split=never_split,
            max_input_chars_per_word=max_input_chars_per_word,
            **kwargs)

    @property
    def special_tokens(self):
        return [(self.pad_token, self.pad_id), (self.unk_token, self.unk_id),
                (self.bos_token, self.bos_id), (self.eos_token, self.eos_id),
                (self.cls_token, self.cls_id), (self.sep_token, self.sep_id),
                (self.mask_token, self.mask_id)]
