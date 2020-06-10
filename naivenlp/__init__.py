import logging

from .tokenizers.abstract_tokenizer import (AbstractTokenizer,
                                            VocabBasedTokenizer)
from .tokenizers.jieba_tokenizer import JiebaTokenizer

logging.basicConfig(
    format="%(asctime)s %(levelname)7s %(filename)15s %(lineno)4d] %(message)s",
    level=logging.INFO
)
