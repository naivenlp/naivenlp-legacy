import os

import pycorrector

from .abstract_corrector import AbstractCorrector


class KenLMCorrector(AbstractCorrector):

    def __init__(self,
                 kenlm_model_path=None,
                 word_freq_files=None,
                 confusion_files=None,
                 same_pinyin_files=None,
                 same_stroke_files=None,
                 stop_word_files=None,
                 ):
        super().__init__()
        if kenlm_model_path:
            self.corrector = pycorrector.Corrector(language_model_path=kenlm_model_path)
        else:
            self.corrector = pycorrector.Corrector()
        self.corrector.check_corrector_initialized()
        self.chinese_char_set = self.corrector.cn_char_set
        self.same_pinyin_map = self.corrector.same_pinyin
        self.same_stroke_map = self.corrector.same_stroke
        self.confusion_map = self.corrector.custom_confusion
        self.word_freq_map = self.corrector.word_freq
        self.stop_words_map = self.corrector.stopwords

        if word_freq_files:
            for f in word_freq_files:
                if not os.path.exists(f):
                    continue
                self.corrector.set_custom_word(f)

        if confusion_files:
            for f in confusion_files:
                if not os.path.exists(f):
                    continue
                self.corrector.set_custom_confusion_dict(f)

        if same_pinyin_files:
            for f in same_pinyin_files:
                if not os.path.exists(f):
                    continue
                m = self.corrector.load_same_pinyin(f, sep='\t')
                self.corrector.same_pinyin.update(m)

        if same_stroke_files:
            for f in same_stroke_files:
                if not os.path.exists(f):
                    continue
                m = self.corrector.load_same_stroke(f, sep='\t')
                self.corrector.same_stroke.update(m)

        if stop_word_files:
            for f in stop_word_files:
                if not os.path.exists(f):
                    continue
                d = self.corrector.load_word_freq_dict(f)
                self.stop_words_map.update(d)

    def correct(self, text, **kwargs):
        corrected, details = self.corrector.correct(text)
        res = []
        for d in details:
            res.append((d[0], d[1], d[2], d[3]))
        return corrected, res
