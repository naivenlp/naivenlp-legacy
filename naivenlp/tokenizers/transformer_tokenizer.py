from .abstract_tokenizer import VocabBasedTokenizer
from .tokenizer import BasicTokenizer, WordpieceTokenizer


class TransformerTokenizer(VocabBasedTokenizer):

    def __init__(self,
                 vocab_file,
                 pad_token='[PAD]',
                 unk_token='[UNK]',
                 bos_token='[BOS]',
                 eos_token='[EOS]',
                 do_lower_case=True,
                 do_basic_tokenization=True,
                 tokenize_chinese_chars=True,
                 never_split=None,
                 max_input_chars_per_word=100,
                 **kwargs):
        super().__init__(
            vocab_file, pad_token=pad_token, unk_token=unk_token, bos_token=bos_token, eos_token=eos_token, **kwargs)
        self.do_lower_case = do_lower_case
        self.do_basic_tokenization = do_basic_tokenization
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.never_split = never_split
        self.max_input_chars_per_word = max_input_chars_per_word

        if self.do_basic_tokenization:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                tokenize_chinese_chars=tokenize_chinese_chars,
                never_split=never_split,
            )
        else:
            self.basic_tokenizer = None

        self.wordpiece_tokenizer = WordpieceTokenizer(
            vocab=set(self.vocab.keys()),
            unk_token=self.unk_token,
            max_input_chars_per_word=self.max_input_chars_per_word)

    def tokenize(self, inputs, never_split=None, **kwargs):
        tokens = []
        never_split = never_split + self.never_split if never_split is not None else self.never_split
        if self.do_basic_tokenization:
            for token in self.basic_tokenizer.tokenize(inputs, never_split=never_split):
                for t in self.wordpiece_tokenizer.tokenize(token):
                    tokens.append(t)
        else:
            tokens = self.wordpiece_tokenizer.tokenize(inputs)

        return tokens
