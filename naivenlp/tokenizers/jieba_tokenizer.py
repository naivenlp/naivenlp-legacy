import logging
import math
import os
import pickle
import re
import time

import pkg_resources

from .abstract_tokenizer import AbstractTokenizer, VocabBasedTokenizer

SPLIT_CHINESE_PATTERN = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\\._%\\-]+)", re.U)
SKIP_PATTERN = re.compile("(\r\n|\\s)", re.U)
ENGLISH_PATTERN = re.compile('[a-zA-Z0-9]', re.U)

get_module_resource = lambda *res: pkg_resources.resource_stream(__name__, os.path.join(*res))


DEFAULT_EMIT_PROB_FILE = "prob_emit.p"
DEFAULT_START_PROB_FILE = "prob_start.p"
DEFAULT_TRANS_PROB_FILE = "prob_trans.p"
DEFAULT_WORDS_FREQ_FILE = "dict.txt"


def count_freq(files, freq_dict, freq_total, sep=' ', default_freq=1, **kwargs):
    for f in files:
        if not os.path.exists(f):
            logging.warning('Load user word freq file failed: %s does not exist. Skipped.', f)
            continue
        with open(f, mode='rt', encoding='utf8') as fin:
            for line in fin:
                parts = line.rstrip('\n').strip().split(sep)
                if not parts:
                    continue
                word = str(parts[0]).strip()
                freq = default_freq if len(parts) <= 1 else int(parts[1])
                freq_dict[word] = freq
                freq_total += freq
                for idx in range(len(word)):
                    c = word[:idx + 1]
                    if c not in freq_dict:
                        freq_dict[c] = 0
        logging.info("Load user dict word freq file successful: %s.", f)
    return freq_dict, freq_total


def count_default_freq(file, sep=" ", default_freq=1, **kwargs):
    words_freq = {}
    words_freq_total = 0
    fd = get_module_resource("data", file)
    for _, line in enumerate(fd, 1):
        line = line.strip().decode("utf-8")
        parts = line.split(sep)
        if not parts:
            continue
        word = str(parts[0]).strip()
        freq = default_freq if len(parts) <= 1 else int(parts[1])
        words_freq[word] = freq
        words_freq_total += freq
        for idx in range(len(word)):
            c = word[:idx+1]
            if c not in words_freq:
                words_freq[c] = 0
    fd.close()
    return words_freq, words_freq_total


def save_words_freq(file, word_freq, **kwargs):
    if not word_freq:
        return
    with open(file, mode='wt', encoding='utf8') as fout:
        for k, v in sorted(word_freq.items(), key=lambda x: x[0]):
            fout.write(k + ' ' + str(v) + '\n')
    logging.info('Words freq dict saved to %s', file)


def load_probs(file):
    if not file:
        return {}
    # if not os.path.exists(file):
    #     return {}
    # with open(file, mode='rb') as fin:
    #     return pickle.load(fin)
    # return {}
    fd = get_module_resource("data", file)
    res = pickle.load(fd)
    fd.close()
    return res


def load_force_split_files(files):
    words = set()
    for f in files:
        if not os.path.exists(f):
            logging.warning('Load force split words file failed: %s does not exists. Skipped.', f)
            continue
        with open(f, mode='rt', encoding='utf8') as fin:
            for line in fin:
                word = line.strip('\n').strip()
                if not word:
                    continue
                words.add(word)
        logging.info('Load force split words file successfully: %s.', f)
    return words


def now():
    return int(round(time.time() * 1000))


_MIN_FLOAT = -3.14e10

PREV_STATUS = {
    'B': 'ES',
    'M': 'MB',
    'S': 'SE',
    'E': 'BM'
}


class HMMTokenizer(object):

    def __init__(self,
                 emit_prob_file=DEFAULT_EMIT_PROB_FILE,
                 start_prob_file=DEFAULT_START_PROB_FILE,
                 trans_prob_file=DEFAULT_TRANS_PROB_FILE,
                 force_split_files=None,
                 **kwargs):
        super().__init__()
        self.has_initialized = False
        self.emit_prob_file = emit_prob_file
        self.emit_prob_dict = None
        self.start_prob_file = start_prob_file
        self.start_prob_dict = None
        self.trans_prob_file = trans_prob_file
        self.trans_prob_dict = None
        self.force_split_files = force_split_files
        self.force_split_words = None

    def initialize(self):
        if self.has_initialized:
            logging.info('Tokenizer has been initialized.')
            return
        s1 = now()
        self.emit_prob_dict = load_probs(self.emit_prob_file)
        logging.info(
            'Load  emit prob file from  %s %s. Cost %d ms.', self.emit_prob_file,
            "successfully" if self.emit_prob_dict else "failed", now() - s1)

        s2 = now()
        self.start_prob_dict = load_probs(self.start_prob_file)
        logging.info(
            'Load start prob file from %s %s. Cost %d ms.', self.start_prob_file,
            "successfully" if self.start_prob_dict else "failed", now() - s2)

        s3 = now()
        self.trans_prob_dict = load_probs(self.trans_prob_file)
        logging.info(
            'Load trans prob file from %s %s. Cost %d ms.', self.trans_prob_file,
            "successfully" if self.trans_prob_dict else "failed", now() - s3)

        if self.force_split_files:
            s4 = now()
            self.force_split_words = load_force_split_files(self.force_split_files)
            logging.info('Load force split words files from %s finished. Cost %d ms.',
                         self.force_split_files, now() - s4)
        else:
            self.force_split_words = set()
            logging.info('Load force split words files skipped. Because no force split files set.')

        logging.info('HMM initialized. Cost %d ms in total.', now() - s1)
        self.has_initialized = True

    def _viterbi(self, obsrvs, states, start_probs, trans_probs, emit_probs):
        V = [{}]
        path = {}
        for y in states:
            V[0][y] = start_probs[y] + emit_probs[y].get(obsrvs[0], _MIN_FLOAT)
            path[y] = [y]

        for x in range(1, len(obsrvs)):
            V.append({})
            new_path = {}
            for y in states:
                emit_p = emit_probs[y].get(obsrvs[x], _MIN_FLOAT)
                (prob, state) = max(
                    [(V[x-1][y0] + trans_probs[y0].get(y, _MIN_FLOAT) + emit_p, y0) for y0 in PREV_STATUS[y]]
                )
                V[x][y] = prob
                new_path[y] = path[state] + [y]
            path = new_path

        prob, state = max(
            (V[len(obsrvs) - 1][y], y) for y in "ES"
        )
        return prob, path[state]

    def _cut(self, sequence, **kwargs):
        prob, positions = self._viterbi(
            sequence, 'BMES', self.start_prob_dict, self.trans_prob_dict, self.emit_prob_dict)
        b, n = 0, 0
        for i, char in enumerate(sequence):
            pos = positions[i]
            if pos == "B":
                b = i
            elif pos == "E":
                yield sequence[b:i+1]
            elif pos == "S":
                yield char
                n = i + 1
        if n < len(sequence):
            yield sequence[n:]

    def tokenize(self, inputs, **kwargs):
        if not inputs:
            return []
        blocks = SPLIT_CHINESE_PATTERN.split(inputs)
        for blk in blocks:
            if SPLIT_CHINESE_PATTERN.match(blk):
                for word in self._cut(blk):
                    if word not in self.force_split_words:
                        yield word
                    else:
                        for w in word:
                            yield w
            else:
                for word in SKIP_PATTERN.split(blk):
                    if word:
                        yield word


class JiebaTokenizer(VocabBasedTokenizer):

    def __init__(self,
                 vocab_file,
                 dict_files=None,
                 emit_prob_file=DEFAULT_EMIT_PROB_FILE,
                 start_prob_file=DEFAULT_START_PROB_FILE,
                 trans_prob_file=DEFAULT_TRANS_PROB_FILE,
                 force_split_files=None,
                 **kwargs):
        super(JiebaTokenizer, self).__init__(
            vocab_file=vocab_file, **kwargs)

        self.default_dict_file = DEFAULT_WORDS_FREQ_FILE
        self.user_dict_files = dict_files

        self.words_freq = {}
        self.words_freq_total = 0

        self.emit_prob_file = emit_prob_file
        self.emit_prob_dict = None
        self.start_prob_file = start_prob_file
        self.start_prob_dict = None
        self.trans_prob_file = trans_prob_file
        self.trans_prob_dict = None

        self.hmm = HMMTokenizer(
            emit_prob_file=emit_prob_file, start_prob_file=start_prob_file,
            trans_prob_file=trans_prob_file, force_split_files=force_split_files)

        self.has_initialized = False

    def add_user_dict(self, dict):
        self.user_dict_files.append(dict)

    def initialize(self, **kwargs):
        if self.has_initialized:
            return
        s0 = now()
        self.words_freq, self.words_freq_total = count_default_freq(self.default_dict_file, **kwargs)
        logging.info('Load default words freq file finished. Cost %d ms.', now() - s0)

        if self.user_dict_files:
            s1 = now()
            count_freq(self.user_dict_files, self.words_freq, self.words_freq_total)
            logging.info('Load user word freq files finished. Cost %d ms.', now()-s1)

        self.hmm.initialize()

        logging.info('Jieba tokenizer initialized. Cost %d ms in total.', now() - s0)
        self.has_initialized = True

    def _build_DAG(self, sequence, **kwargs):
        if not self.has_initialized:
            self.initialize(**kwargs)
        DAG = {}
        N = len(sequence)
        for k in range(N):
            ll = []
            i = k
            seg = sequence[k]
            while i < N and seg in self.words_freq:
                if self.words_freq[seg]:
                    ll.append(i)
                i += 1
                seg = sequence[k:i + 1]
            if not ll:
                ll.append(k)
            DAG[k] = ll
        return DAG

    def _calc_route(self, sequence, dag):
        route = {}
        N = len(sequence)
        route[N] = (0, 0)
        log_freq = math.log(self.words_freq_total)
        for idx in range(N - 1, -1, -1):
            route[idx] = max(
                (math.log(self.words_freq.get(sequence[idx: x + 1], 1) - log_freq + route[x + 1][0]), x) for x in
                dag[idx]
            )
        return route

    def _cut_all(self, sequence, **kwargs):
        dag = self._build_DAG(sequence, **kwargs)
        idx = -1
        english_scan, english_buffer = 0, ''
        for k, l in dag.items():
            if english_scan == 1 and ENGLISH_PATTERN.match(sequence[k]):
                english_scan = 0
                yield english_buffer
            if len(l) == 1 and k > idx:
                word = sequence[k:l[0] + 1]
                if ENGLISH_PATTERN.match(word):
                    if english_scan == 0:
                        english_scan = 1
                        english_buffer = word
                    else:
                        english_buffer += word
                if english_scan == 0:
                    yield word
                idx = l[0]
            else:
                for j in l:
                    if j > k:
                        yield sequence[k:j + 1]
                        idx = j

        if english_scan == 1:
            yield english_buffer

    def _cut_hmm(self, sequence, **kwargs):
        dag = self._build_DAG(sequence)
        route = self._calc_route(sequence, dag)
        x = 0
        N = len(sequence)
        buffer = ''
        while x < N:
            y = route[x][1] + 1
            word = sequence[x:y]
            if y-x == 1:
                buffer += word
                x = y
                continue
            if buffer:
                if len(buffer) == 1:
                    yield buffer
                    buffer = ''
                else:
                    if not self.words_freq.get(buffer, 0):
                        for t in self.hmm.tokenize(buffer):
                            yield t
                    else:
                        for t in buffer:
                            yield t
                    buffer = ''

            yield word
            x = y

        if buffer:
            if len(buffer) == 1:
                yield buffer
            elif not self.words_freq.get(buffer, 0):
                for t in self._hmm_decode(buffer):
                    yield t
            else:
                for t in buffer:
                    yield t

    def _cut_default(self, sequence, **kwargs):
        dag = self._build_DAG(sequence)
        route = self._calc_route(sequence, dag)
        start = 0
        N = len(sequence)
        buffer = ''
        while start < N:
            end = route[start][1] + 1
            word = sequence[start:end]
            if ENGLISH_PATTERN.match(word) and len(word) == 1:
                buffer += word
                start = end
            else:
                if buffer:
                    yield buffer
                    buffer = ''
                yield word
                start = end
        if buffer:
            yield buffer
            buffer = ''

    def cut(self, sequence, cut_all=False, hmm=True, **kwargs):
        if not sequence:
            return []

        blocks = SPLIT_CHINESE_PATTERN.split(sequence)
        if not blocks:
            return []

        if cut_all:
            cut_fn = self._cut_all
        elif hmm:
            cut_fn = self._cut_hmm
        else:
            cut_fn = self._cut_default

        for blk in blocks:
            if not blk:
                continue
            if SPLIT_CHINESE_PATTERN.match(blk):
                for word in cut_fn(blk):
                    yield word
            else:
                segments = SKIP_PATTERN.split(blk)
                for seg in segments:
                    if SKIP_PATTERN.match(seg):
                        yield seg
                    elif not cut_all:
                        for s in seg:
                            yield s
                    else:
                        yield seg

    def tokenize(self, inputs, cut_all=False, hmm=True, **kwargs):
        tokens = [t for t in self.cut(inputs, cut_all=cut_all, hmm=hmm, **kwargs)]
        return tokens
