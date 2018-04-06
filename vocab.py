
import json
import pickle
import codecs
import numpy as np
import random

class Vocabulary:
    def __init__(self):
        self.hz_2_i, self.i_2_hz = pickle.load(open('../dicts/hz-vocab.pkl', 'rb'))
        self.w_2_i = pickle.load(open('../dicts/w2v.pkl', 'rb'))
        self.vocab_size = len(self.hz_2_i)
        self.output_dim = len(self.hz_2_i)

    def index_to_hz(self, index):
        return self.i_2_hz[index]

    def get_vocab_size(self):
        return self.vocab_size

    def get_output_dim(self):
        return self.output_dim

    def get_hz_index(self, hz):
        return self.hz_2_i.get(hz, 1)

    def get_word_index(self, hz):
        return self.w_2_i.get(hz,1)

    def encode_output(self, y):
        r = [0] * self.output_dim
        r[y] = 1
        return r

    def encode_y_output(self, y):
        from keras.utils.np_utils import to_categorical
        return to_categorical(y, self.output_dim)

    def encode_batch_output(self, ys):
        return np.array([self.encode_y_output(y) for y in ys])
