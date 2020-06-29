import abc
import logging
import os
from typing import List

from naivenlp.tokenizers.abstract_tokenizer import AbstractTokenizer

from .detector import AbstractDetector, KenLMDetector

pwd = os.path.abspath(os.path.dirname(__file__))


DEFAULT_WORD_FREQ_FILES = [
    'data/word_freq.txt',
    'data/custom_word_freq.txt',
    'data/place_name.txt',
    'data/person_name.txt',
]
DEFAULT_WORD_FREQ_FILES = [os.path.join(pwd, d) for d in DEFAULT_WORD_FREQ_FILES]

DEFAULT_CONFUSION_FILES = [
    'data/custom_confusion.txt'
]
DEFAULT_CONFUSION_FILES = [os.path.join(pwd, d) for d in DEFAULT_CONFUSION_FILES]

DEFAULT_STOP_WORD_FILES = [
    'data/stopwords.txt',
]
DEFAULT_STOP_WORD_FILES = [os.path.join(pwd, d) for d in DEFAULT_STOP_WORD_FILES]

DEFAULT_SAME_PINYIN_FILES = [
    'data/same_pinyin.txt',
]
DEFAULT_SAME_PINYIN_FILES = [os.path.join(pwd, d) for d in DEFAULT_SAME_PINYIN_FILES]

DEFAULT_SAME_STROKE_FILES = [
    'data/same_stroke.txt'
]
DEFAULT_SAME_STROKE_FILES = [os.path.join(pwd, d) for d in DEFAULT_SAME_STROKE_FILES]


class AbstractCorrector(abc.ABC):

    def correct(self, text, **kwargs):
        raise NotImplementedError()


class AbstractCorrectorCallback:

    def on_start(self, text):
        pass

    def on_detect_start(self, text):
        pass

    def on_detect_end(self, errors):
        pass

    def on_correct_start(self, errors):
        pass

    def on_correct_end(self, errors, corrections):
        pass

    def on_end(self, corrected_text):
        pass


class CorrectorCallbackWrapper(AbstractCorrectorCallback):

    def __init__(self, callbacks: List[AbstractCorrectorCallback] = None):
        super().__init__()
        self.callbacks = callbacks

    def on_start(self, text):
        if not self.callbacks:
            return
        for cb in self.callbacks:
            if not cb:
                continue
            cb.on_start(text)

    def on_detect_start(self, text):
        if not self.callbacks:
            return
        for cb in self.callbacks:
            if not cb:
                continue
            cb.on_detect_start(text)

    def on_detect_end(self, errors):
        if not self.callbacks:
            return
        for cb in self.callbacks:
            if not cb:
                continue
            cb.on_detect_end(errors)

    def on_correct_start(self, errors):
        if not self.callbacks:
            return
        for cb in self.callbacks:
            if not cb:
                continue
            cb.on_correct_start(errors)

    def on_correct_end(self, errors, corrections):
        if not self.callbacks:
            return
        for cb in self.callbacks:
            if not cb:
                continue
            cb.on_correct_end(errors, corrections)

    def on_end(self, corrected_text):
        if not self.callbacks:
            return
        for cb in self.callbacks:
            if not cb:
                continue
            cb.on_end(corrected_text)


class KenLMCorrector(AbstractCorrector):

    def __init__(self,
                 kenlm_model_path=None,
                 word_freq_files=None,
                 confusion_files=None,
                 stop_word_files=None,
                 same_pinyin_files=None,
                 same_stroke_files=None,
                 tokenizer: AbstractTokenizer = None,
                 **kwargs):
        super().__init__()

        self.word_freq_map = {}
        if not word_freq_files:
            word_freq_files = []
        word_freq_files.extend(DEFAULT_WORD_FREQ_FILES)
        self._load_word_freq_files(word_freq_files)
        logging.info('Load word frequency files finished.')

        self.confusion_map = {}
        if not confusion_files:
            confusion_files = []
        confusion_files.extend(DEFAULT_CONFUSION_FILES)
        self._load_confusion_files(confusion_files)
        logging.info('Load confusion files finished.')

        self.stop_words = set()
        if not stop_word_files:
            stop_word_files = []
        stop_word_files.extend(DEFAULT_STOP_WORD_FILES)
        self._load_stop_word_files(stop_word_files)
        logging.info('Load stop word files finished.')

        self.same_pinyin_chars = {}
        if not same_pinyin_files:
            same_pinyin_files = []
        same_pinyin_files.extend(DEFAULT_SAME_PINYIN_FILES)
        self._load_same_pinyin_files(same_pinyin_files)
        logging.info('Load same pinyin files finished.')

        self.same_stroke_chars = {}
        if not same_stroke_files:
            same_stroke_files = []
        same_stroke_files.extend(DEFAULT_SAME_STROKE_FILES)
        self._load_same_stroke_files(same_stroke_files)
        logging.info('Load same stroke files finished.')

        self.detector = KenLMDetector(
            kenlm_model_path=kenlm_model_path,
            confusion_map=self.confusion_map,
            word_freq_map=self.word_freq_map,
            stop_words=self.stop_words,
            tokenizer=tokenizer)
        logging.info('Initialized detector finished.')

    def _get_same_pinyin_candidates_by_word(self, word):
        candidates = set()

        same_pinyin_items = []
        for w in word:
            item = list(self.same_pinyin_map.get(w, set([w])))
            same_pinyin_items.append(item)

        def _backtracking(arrs, idx, ans):
            if len(ans) == len(arrs):
                if ans in self.word_freq_map:
                    candidates.add(ans)
                return
            arr = arrs[idx]
            for i in range(len(arr)):
                arrs[idx] = arr[:i] + arr[i + 1:]
                _backtracking(arrs, idx + 1, ans + arr[i])
                arrs[idx] = arr

        _backtracking(same_pinyin_items, 0, '')

        return candidates

    def _get_same_pinyin_candidates_by_char(self, word):
        candidates = set()
        for i in range(len(word)):
            prefix = word[:i]
            suffix = word[i + 1:] if i < len(word) - 1 else ''
            w = word[i]
            chars = self.same_pinyin_chars.get(w, None)
            if not chars:
                continue
            for c in chars:
                cand = prefix + c + suffix
                if cand in self.detector.word_freq_map:
                    candidates.add(cand)
        return candidates

    def _get_same_stroke_candidates_by_char(self, word):
        candidates = set()
        for i in range(len(word)):
            prefix = word[:i]
            suffix = word[i + 1:] if i < len(word) - 1 else ''
            w = word[i]
            if w in self.same_stroke_chars:
                chars = self.same_stroke_chars[w]
                for c in chars:
                    cand = prefix + c + suffix
                    if cand in self.detector.word_freq_map:
                        candidates.add(cand)
        return candidates

    def _get_same_stroke_candidates_by_word(self, word):
        candidates = set()

        same_stroke_items = []
        for w in word:
            item = list(self.same_stroke_chars.get(w, set([w])))
            same_stroke_items.append(item)

        def _backtracking(arrs, idx, ans):
            if len(ans) == len(arrs):
                if ans in self.word_freq_map:
                    candidates.add(ans)
                return

            arr = arrs[idx]
            for i in range(len(arr)):
                arrs[idx] = arr[:i] + arr[i + 1:]
                _backtracking(arrs, idx + 1, ans + arr[i])
                arrs[idx] = arr

        _backtracking(same_stroke_items, 0, '')

        return candidates

    def _generate_candidates(self, token, topn=50):
        candidates = set()
        candidates.add(token)
        candidates.union(self._get_same_pinyin_candidates_by_word(token))
        candidates.union(self._get_same_pinyin_candidates_by_char(token))
        candidates.union(self._get_same_stroke_candidates_by_word(token))
        candidates.union(self._get_same_stroke_candidates_by_char(token))
        candidates = sorted(list(candidates), key=lambda x: self.word_freq_map.get(x, 0), reverse=True)
        candidates = candidates[:topn]
        return candidates

    def _select_candidate(self, token, candidates):
        pass

    def correct(self,
                text,
                keep_symbol=False,
                detect_word=False,
                detect_char=True,
                ngrams=[2, 3],
                ratio=0.6745,
                threshold=2,
                callbacks: List[AbstractCorrectorCallback] = None,
                topn=50,
                return_details=False,
                **kwargs):
        callback = CorrectorCallbackWrapper(callbacks)
        callback.on_start(text)

        callback.on_detect_start(text)
        errors = self.detector.detect(
            text, keep_symbol=keep_symbol, detect_word=detect_word, detect_char=detect_char,
            ngrams=ngrams, ratio=ratio, threshold=threshold, **kwargs)
        callback.on_detect_end(errors)

        callback.on_correct_start(errors)
        corrections = []
        corrected_text = ''
        for err in errors:
            token, start_idx, end_idx, err_type = err
            if err_type == 'CONFUSION':
                corrected_token = self.confusion_map[token]
                corrections.append((token, corrected_token, start_idx, end_idx))
                corrected_text += corrected_token
                continue
            if err_type in ['CHAR', 'WORD']:
                candidate_tokens = self._generate_candidates(token, topn=topn)
                if candidate_tokens:
                    corrected_token = self._select_candidate(token, candidate_tokens)
                    corrections.append((token, corrected_token, start_idx, end_idx))
                    corrected_text += corrected_token
                else:
                    corrected_text += token
                continue

            corrected_text += token

        callback.on_correct_end(errors, corrections)

        callback.on_end(corrected_text)

        if return_details:
            return corrected_text, corrections

        return corrected_text

    def _load_word_freq_files(self, files):
        if not files:
            return {}
        for f in files:
            if not os.path.exists(f):
                logging.warning('Load word freq file: {} failed. File does not exist. Skipped.'.format(f))
                continue
            with open(f, mode='rt', encoding='utf8') as fin:
                for line in fin:
                    line = line.strip('\n').strip()
                    if not line:
                        continue
                    if line.startswith('#'):
                        continue
                    words = line.split()
                    if len(words) < 2:
                        continue
                    word, freq = words[0], int(words[1])
                    self.word_freq_map[word] = freq
            logging.info('Load word freq file: {} successfully.'.format(f))

    def _load_confusion_files(self, files):
        if not files:
            logging.warning('`confusion_files` is empty or None.')
            return {}
        for f in files:
            if not os.path.exists(f):
                logging.warning('Load confusion file: {} fialed. File does not exist. Skipped.'.format(f))
                continue
            with open(f, mode='rt', encoding='utf8') as fin:
                for line in fin:
                    line = line.strip('\n').strip()
                    if not line:
                        continue
                    if line.startswith('#'):
                        continue
                    words = line.split()
                    if len(words) < 2:
                        continue
                    variant, origin = words[0], words[1]
                    freq = int(words[2]) if len(words) > 2 else 1
                    self.word_freq_map[origin] = max(freq, self.word_freq_map.get(origin, 0))
                    self.confusion_map[variant] = origin
            logging.info('Load confusion file: {} successfully.'.format(f))

    def _load_stop_word_files(self, files):
        if not files:
            logging.warning('`stop_word_files` is empty or None.')
            return {}
        for f in files:
            if not os.path.exists(f):
                logging.warning('Load stop word file: {} failed. File does not exist. Skipped.'.format(f))
                continue
            with open(f, mode='rt', encoding='utf8') as fin:
                for line in fin:
                    line = line.strip('\n').strip()
                    if not line:
                        continue
                    parts = line.split(' ')
                    if len(parts) < 1:
                        continue
                    word = parts[0]
                    freq = int(parts[1]) if len(parts) >= 2 else 1
                    self.stop_words.add(word)
                    self.word_freq_map[word] = max(freq, self.word_freq_map.get(word, 0))
            logging.info('Load stop word file: {} successfully.'.format(f))

    def _load_same_pinyin_files(self, files):
        if not files:
            logging.warning('Load no same pinyin files.')
            return {}
        for f in files:
            if not os.path.exists(f):
                logging.warning('Load same pinyin file: {} failed. File does not exist. Skipped.'.format(f))
                continue
            with open(f, mode='rt', encoding='utf8') as fin:
                for line in fin:
                    line = line.strip('\n').strip()
                    if not line:
                        continue
                    if line.startswith('#'):
                        continue
                    parts = line.split('\t')
                    if len(parts) < 1:
                        continue
                    word = parts[0]
                    vairance = ''.join(parts[1:])
                    same = set()
                    for v in vairance:
                        if not v.strip():
                            continue
                        same.add(v)
                    self.same_pinyin_chars[word] = same.union(self.same_pinyin_chars.get(word, set()))
            logging.info('Load same pinyin file: {} successfully.'.format(f))

    def _load_same_stroke_files(self, files):
        if not files:
            logging.warning('Load no same stroke files.')
            return {}
        for f in files:
            if not os.path.exists(f):
                logging.warning('Load same stroke file: {} failed. File does not exist. Skipped.'.format(f))
                continue
            with open(f, mode='rt', encoding='utf8') as fin:
                for line in fin:
                    line = line.strip('\n').strip()
                    if not line:
                        continue
                    if line.startswith('#'):
                        continue
                    parts = line.split('\t')
                    if len(parts) < 1:
                        continue
                    word = parts[0]
                    vairance = ''.join(parts[1:])
                    same = set()
                    for v in vairance:
                        if not v.strip():
                            continue
                        same.add(v)
                    self.same_stroke_chars[word] = same.union(self.same_stroke_chars.get(word, set()))
            logging.info('Load same stroke file: {} successfully.'.format(f))
