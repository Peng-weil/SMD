import numpy as np
from logging import getLogger

logger = getLogger()


class LLIM_ENV(object):

    def __init__(self, params):
        tokens = ['<s>', '</s>', '<pad>', '0', '1', '2', '3', '4', '5']
        self.words = tokens

        self.id2word = {i: s for i, s in enumerate(self.words)}
        self.word2id = {s: i for i, s in self.id2word.items()}

        self.n_words = params.n_words = len(self.words)
        self.eos_index = params.eos_index = 0
        self.pad_index = params.pad_index = 1

        logger.info(f"word2id: {self.word2id}")