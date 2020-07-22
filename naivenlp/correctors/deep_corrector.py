import os
import logging

import tensorflow as tf

from .abstract_corrector import AbstractCorrector


class DeepCorrector(AbstractCorrector):

    def __init__(self, saved_model, **kwargs):
        loaded = tf.saved_model.load(saved_model)
        logging.info('Load saved model from {} successfully.'.format(saved_model))
        self.model = loaded.signatures['serving_default']

    def correct(self, text, **kwargs):
        raise NotImplementedError()


class TransformerCorrector(DeepCorrector):

    def correct(self, text, **kwargs):
        inputs = tf.constant([' '.join(text[:])], dtype=tf.string)
        outputs = self.model(inputs)
        tokens, probs = outputs['text'].numpy()[0], outputs['log_probs'].numpy()[0]
        tokens = ''.join(''.join([t.decode('utf-8') for t in tokens]).split())
        return tokens, probs[0]
