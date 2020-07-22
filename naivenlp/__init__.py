import logging

from naivenlp.similarity import (
    cosine_distance,
    cosine_similarity,
    damerau_distance,
    jaccard_distance,
    jaccard_similarity,
    lcs_distance,
    lcs_length,
    levenshtein_distance,
    levenshtein_distance_normalized,
    levenshtein_similarity,
    longest_common_subsequence_distance,
    longest_common_subsequence_length,
    optimal_string_alignment_distance,
    osa_distance,
    sorense_dice_distance,
    sorense_dice_similarity,
    weighted_levenshtein_distance,
)
from naivenlp.structures.trie import AbstractTrie, Node, Trie
from naivenlp.tokenizers.abstract_tokenizer import AbstractTokenizer, CustomTokenizer, VocabBasedTokenizer
from naivenlp.tokenizers.bert_tokenizer import BertTokenizer
from naivenlp.tokenizers.tokenizer import BasicTokenizer, WordpieceTokenizer
from naivenlp.tokenizers.transformer_tokenizer import TransformerTokenizer
from naivenlp.utils.texts import b2q, q2b, split_sentence

__version__ = "0.0.9"
__name__ = "naivenlp"


logging.basicConfig(
    format="%(asctime)s %(levelname)7s %(filename)20s %(lineno)4d] %(message)s",
    level=logging.INFO
)
