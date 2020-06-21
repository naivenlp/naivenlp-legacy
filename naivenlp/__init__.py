import logging

from .tokenizers.abstract_tokenizer import AbstractTokenizer, VocabBasedTokenizer
from .tokenizers.jieba_tokenizer import ACCURATE_MODE, FULL_MODE, SEARCH_MODE, JiebaTokenizer
from .tokenizers.language_model_tokenizer import LanguageModelTokenizer

__version__ = "0.0.2"
__name__ = "naivenlp"


logging.basicConfig(
    format="%(asctime)s %(levelname)7s %(filename)20s %(lineno)4d] %(message)s",
    level=logging.INFO
)
