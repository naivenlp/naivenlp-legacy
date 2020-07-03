from strsimpy import (
    Cosine,
    Damerau,
    Jaccard,
    Levenshtein,
    LongestCommonSubsequence,
    NormalizedLevenshtein,
    OptimalStringAlignment,
    SorensenDice,
    WeightedLevenshtein,
)
from strsimpy.weighted_levenshtein import default_deletion_cost, default_insertion_cost, default_substitution_cost

DEFAULT_K = 3
DEFAULT_COSINE = Cosine(k=DEFAULT_K)
DEFAULT_JACCARD = Jaccard(k=DEFAULT_K)
DEFAULT_LEVENSHTEIN = Levenshtein()
DEFAULT_NORMALIZED_LEVENSHTEIN = NormalizedLevenshtein()
DEFAULT_WEIGHTED_LEVENSHTEIN = WeightedLevenshtein()
DEFAULT_DAMERAU = Damerau()
DEFAULT_LCS = LongestCommonSubsequence()
DEFAULT_SORENSEN_DICE = SorensenDice(k=DEFAULT_K)
DEFAULT_OSA = OptimalStringAlignment()


def cosine_distance(a, b, k=DEFAULT_K):
    cosine = DEFAULT_COSINE if k == DEFAULT_K else Cosine(k=k)
    return cosine.distance(a, b)


def cosine_similarity(a, b, k=DEFAULT_K):
    cosine = DEFAULT_COSINE if k == DEFAULT_K else Cosine(k=k)
    return cosine.distance(a, b)


def jaccard_distance(a, b, k=DEFAULT_K):
    jaccard = DEFAULT_JACCARD if k == DEFAULT_K else Jaccard(k=k)
    return jaccard.distance(a, b)


def jaccard_similarity(a, b, k=DEFAULT_K):
    jaccard = DEFAULT_JACCARD if k == DEFAULT_K else Jaccard(k=k)
    return jaccard.similarity(a, b)


def levenshtein_distance(a, b):
    return DEFAULT_LEVENSHTEIN.distance(a, b)


def levenshtein_distance_normalized(a, b):
    return DEFAULT_NORMALIZED_LEVENSHTEIN.distance(a, b)


def levenshtein_similarity(a, b):
    return DEFAULT_NORMALIZED_LEVENSHTEIN.similarity(a, b)


def weighted_levenshtein_distance(a, b, substitution_cost_fn=None, insertion_cost_fn=None, deletion_cost_fn=None):
    substitution_fn = substitution_cost_fn or default_substitution_cost
    insertion_fn = insertion_cost_fn or default_insertion_cost
    deletion_fn = deletion_cost_fn or default_deletion_cost
    wl = WeightedLevenshtein(
        substitution_cost_fn=substitution_fn,
        insertion_cost_fn=insertion_fn,
        deletion_cost_fn=deletion_fn)
    return wl.distance(a, b)


def damerau_distance(a, b):
    return DEFAULT_DAMERAU.distance(a, b)


def lcs_distance(a, b):
    return DEFAULT_LCS.distance(a, b)


def lcs_length(a, b):
    return DEFAULT_LCS.length(a, b)


def longest_common_subsequence_distance(a, b):
    return DEFAULT_LCS.distance(a, b)


def longest_common_subsequence_length(a, b):
    return DEFAULT_LCS.length(a, b)


def sorense_dice_distance(a, b, k=DEFAULT_K):
    sorense_dice = DEFAULT_SORENSEN_DICE if k == DEFAULT_K else SorensenDice(k=k)
    return sorense_dice.distance(a, b)


def sorense_dice_similarity(a, b, k=DEFAULT_K):
    sorense_dice = DEFAULT_SORENSEN_DICE if k == DEFAULT_K else SorensenDice(k=k)
    return sorense_dice.similarity(a, b)


def osa_distance(a, b):
    return DEFAULT_OSA.distance(a, b)


def optimal_string_alignment_distance(a, b):
    return DEFAULT_OSA.distance(a, b)
