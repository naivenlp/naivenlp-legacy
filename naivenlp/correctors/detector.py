import abc
import logging
import re

import kenlm
import numpy as np

from naivenlp.tokenizers.abstract_tokenizer import AbstractTokenizer
from naivenlp.utils import texts
from naivenlp.utils.get_file import get_file

CHINESE_PATTERN = re.compile(r'([\u4E00-\u9FD5a-zA-Z0-9+#&]+)', re.U)

KEN_LM_MODEL_PEOPLE_CHARS_LM = 'people_chars_lm.klm'

KEN_LM_MODEL_URLS = {
    KEN_LM_MODEL_PEOPLE_CHARS_LM: 'https://www.borntowin.cn/mm/emb_models/people_chars_lm.klm'
}


class AbstractDetector(abc.ABC):

    def detect(self, text, **kwargs):
        raise NotImplementedError()


class KenLMDetector(AbstractDetector):

    def __init__(self,
                 kenlm_model_path=None,
                 confusion_map=None,
                 word_freq_map=None,
                 stop_words=None,
                 tokenizer: AbstractTokenizer = None):
        super().__init__()
        if not kenlm_model_path:
            logging.info('kenlm_model_path not specified, use the default model {} from {}'.format(
                KEN_LM_MODEL_PEOPLE_CHARS_LM, KEN_LM_MODEL_URLS[KEN_LM_MODEL_PEOPLE_CHARS_LM]))
            kenlm_model_path = get_file(
                fname=KEN_LM_MODEL_PEOPLE_CHARS_LM,
                origin=KEN_LM_MODEL_URLS[KEN_LM_MODEL_PEOPLE_CHARS_LM],
                cache_dir='~/.naivenlp',
                cache_subdir='correctors',
            )
        self.kenlm_model = kenlm.Model(kenlm_model_path)
        self.confusion_map = confusion_map if confusion_map is not None else {}
        self.word_freq_map = word_freq_map if word_freq_map is not None else {}
        self.stop_words = stop_words if stop_words is not None else set()
        self.tokenizer = tokenizer

    def _segment(self, text, keep_symbol=False, **kwargs):
        segements = []
        blocks = CHINESE_PATTERN.split(text)
        idx = 0
        for blk in blocks:
            if not blk.strip():
                continue
            if keep_symbol:
                segements.append((blk, idx))
            elif CHINESE_PATTERN.match(blk):
                segements.append((blk, idx))
            idx += len(blk)
        return segements

    def _maybe_error_index(self, scores, ratio=0.6745, threshold=2):
        result = []
        scores = np.array(scores)
        if len(scores.shape) == 1:
            scores = scores[:, np.newaxis]
        median = np.median(scores, axis=0)
        margin_median = np.abs(scores - median).flatten()
        median_abs_deviation = np.median(margin_median)
        if median_abs_deviation == 0:
            return result
        y_score = ratio * margin_median / median_abs_deviation
        scores = scores.flatten()
        maybe_err_inx = np.where((y_score > threshold) & (scores < median))
        result = list(maybe_err_inx[0])
        return result

    def _filter_token(self, token: str):
        if not token.strip():
            return True
        if token.isdigit():
            return True
        if all('a' < c < 'z' for c in token.lower()):
            return True
        if not re.match(r'^[\u4E00-\u9FD5]+$', token):
            return True
        return False

    def _detect_word(self, segment, start_idx):
        if not self.tokenizer:
            raise ValueError("You must provide tokenizer when detecting word errors.")
        maybe_errors = []
        tokens = self.tokenizer.tokenize(segment)
        idx = 0
        for token in tokens:
            if token in self.word_freq_map:
                continue
            if self._filter_token(token):
                continue
            begin_idx = idx
            end_idx = begin_idx + len(token)
            maybe_err = (token, start_idx + begin_idx, start_idx + end_idx, 'WORD')
            maybe_errors.append(maybe_err)
        return maybe_errors

    def _detect_char(self, segment, start_idx, ngrams=[2, 3], ratio=0.6745, threshold=2):
        maybe_errors = []
        ngram_avg_scores = []
        for n in ngrams:
            scores = []
            for i in range(len(segment) - n + 1):
                ngram = segment[i:i + n]
                s = self.kenlm_model.score(' '.join(ngram), bos=False, eos=False)
                scores.append(s)
            if not scores:
                continue
            for _ in range(n - 1):
                scores.insert(0, scores[0])
                scores.append(scores[-1])

            avg_score = [sum(scores[i: i + n]) / len(scores[i: i + n]) for i in range(len(segment))]
            ngram_avg_scores.append(avg_score)

        if len(ngram_avg_scores) > 0:
            segment_scores = list(np.average(np.array(ngram_avg_scores), axis=0))
            for i in self._maybe_error_index(segment_scores, ratio=ratio, threshold=threshold):
                token = segment[i]
                if not token:
                    continue
                if token in self.stop_words:
                    continue
                if self._filter_token(token):
                    continue
                maybe_err = (token, i + start_idx, i + start_idx + 1, 'CHAR')
                maybe_errors.append(maybe_err)

        return maybe_errors

    def _detect(self,
                segment,
                start_idx,
                detect_word=False,
                detect_char=True,
                ngrams=[2, 3],
                ratio=0.6745,
                threshold=2,
                **kwargs):
        maybe_errors = []
        for confusion in self.confusion_map.keys():
            idx = segment.find(confusion)
            if idx >= 0:
                maybe_err = (confusion, start_idx + idx, start_idx + idx + len(confusion), 'CONFUSION')
                maybe_errors.append(maybe_err)
                return maybe_errors

        if detect_word:
            maybe_errors.extend(self._detect_word(segment, start_idx))
        if detect_char:
            maybe_errors.extend(self._detect_char(segment, start_idx, ngrams, ratio, threshold))

        return maybe_errors

    def detect(self,
               text,
               keep_symbol=False,
               detect_word=False,
               detect_char=True,
               ngrams=[2, 3],
               ratio=0.6745,
               threshold=2,
               **kwargs):
        if not text or not text.strip():
            return []
        text = texts.string_full2half(text)
        text = text.lower()

        segments = self._segment(text, keep_symbol=keep_symbol, **kwargs)
        res = []
        for s, idx in segments:
            for e in self._detect(
                    s,
                    start_idx=idx, detect_word=detect_word, detect_char=detect_char,
                    ngrams=ngrams, ratio=ratio, threshold=threshold, **kwargs):
                res.append(e)
        return res
