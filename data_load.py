
'''
Data loading.
Note:
Nine key pinyin keyboard layout sample:

`      ABC   DEF
GHI    JKL   MNO
POQRS  TUV   WXYZ

'''
from __future__ import print_function
from hyperparams import Hyperparams as hp
import codecs
import numpy as np
import re
import glob
import tensorflow as tf
import random

def load_vocab():
    import pickle
    hz_2_i, i_2_hz = pickle.load(open('../dicts/hz-vocab.pkl', 'rb'))
    py_2_i, i_2_py = pickle.load(open('../dicts/py-vocab.pkl', 'rb'))

    return py_2_i, i_2_py, hz_2_i, i_2_hz

def load_vocab_json():
    import json
    hz_2_i, i_2_hz = json.load(open('../dicts/hz-vocab.json', 'r'))
    py_2_i, i_2_py = json.load(open('../dicts/py-vocab.json', 'r'))

    return py_2_i, i_2_py, hz_2_i, i_2_hz

def load_embeddings():
    import json
    _, _, embeddings = json.load(open('../dicts/w2v.json', 'r'))
    return embeddings

def load_w2v():
    import json
    w2i, i2w, _ = json.load(open('../dicts/w2v.json', 'r'))
    return w2i, i2w


train_data_files = []
train_data_files += glob.glob("../data/simplify/train/train.*.tfrecords")

test_data_files = []
#test_data_files += glob.glob("../data/all/test/train.*.tfrecords")
#test_data_files += glob.glob("../data/gudian/test/train.*.tfrecords")
test_data_files += glob.glob("../data/simplify/test/train.*.tfrecords")
#test_data_files += glob.glob("../data/lishi/test/train.*.tfrecords")
#test_data_files += glob.glob("../data/wuxia/test/train.*.tfrecords")
#test_data_files += glob.glob("../data/qingchun/test/train.*.tfrecords")
############################################################
def parse(example, start_index, eof_index):
   features = tf.parse_single_example(example,
                                   features = {
                                       'x' : tf.FixedLenFeature([], tf.string),
                                       'y' : tf.FixedLenFeature([], tf.string)
                                   })

   x =  tf.decode_raw(features['x'], tf.int32)
   y =  tf.decode_raw(features['y'], tf.int32)

   yhat = tf.concat([[start_index], y], axis=-1)
   ylabel = tf.concat([y, [eof_index]], axis=-1)

   return x, yhat, ylabel, tf.shape(yhat)

############################################################
def load_dataset(batch_size, start_index, eof_index):
   random.shuffle(train_data_files)
   dataset = tf.data.TFRecordDataset(train_data_files)
   dataset = dataset.map(lambda x: parse(x, start_index, eof_index), num_parallel_calls=8).shuffle(2560)
   dataset = dataset.padded_batch(batch_size, padded_shapes=([None], [None], [None], [1])).prefetch(4096).repeat()
   return dataset

def load_test_dataset(batch_size, start_index, eof_index):
   dataset = tf.data.TFRecordDataset(test_data_files)
   dataset = dataset.map(lambda x: parse(x, start_index, eof_index))
   #dataset = dataset.filter(lambda a,b,c: tf.shape(a)[-1] > 5)
   dataset = dataset.shuffle(1280)
   dataset = dataset.padded_batch(batch_size, padded_shapes=([None], [None], [None], [1])).repeat()
   return dataset

from pypinyin import pinyin, Style
import pypinyin

def is_alpha(word):
  try:
    return word.encode('ascii').isalpha()
  except:
    return False

#pinyin(c, style=Style.NORMAL)
def transform_char(c):
    if is_alpha(c):
        return '@' + c.lower()
    elif c.isspace():
        return '@@'
    else:
        return c

def trans(str):
    return [transform_char(s) for s in str]

def same_pinyin(w1, w2):
    p1 = pinyin(w1, style=Style.NORMAL, heteronym=True, errors='ignore')
    p2 = pinyin(w2, style=Style.NORMAL, heteronym=True, errors='ignore')
    if len(p1) == 0 or len(p2) == 0:
        return False

    ret = list(set(p1[0]).intersection(set(p2[0])))

    return True if len(ret) > 0 else False

def to_pinyin(sentence):
    sent = pinyin(sentence, style=Style.NORMAL, errors=trans)
    result = []
    for s in sent:
        result += s

    return " ".join(result)

def load_test_string(w2idx, test_string):
    '''Embeds and vectorize words in user input string'''
    #print(pnyn_sent)
    xs = []
    #lens = []
    x = [w2idx.get(w, 1) for w in test_string]
    #x += [0] * (hp.maxlen - len(x))
    xs.append(x)
    #lens.append([len(x)])

    X = np.array(xs, np.int32)
    #L = np.array(lens, np.int32)

    return X#, L


# def get_batch():
#     '''Makes batch queues from the training data.
#     Returns:
#       A Tuple of x (Tensor), y (Tensor).
#       x and y have the shape [batch_size, maxlen].
#     '''
#     import tensorflow as tf
#
#     # Load data
#     X, Y = load_train_data()
#
#     # Create Queues
#     x, y = tf.train.slice_input_producer([tf.convert_to_tensor(X),
#                                           tf.convert_to_tensor(Y)])
#
#     x = tf.decode_raw(x, tf.int32)
#     y = tf.decode_raw(y, tf.int32)
#
#     x, y = tf.train.batch([x, y],
#                           shapes=[(None,), (None,)],
#                           num_threads=30,
#                           batch_size=hp.batch_size,
#                           capacity=hp.batch_size * 64,
#                           allow_smaller_final_batch=False,
#                           dynamic_pad=True)
#     num_batch = len(X) // hp.batch_size
#
#     return x, y, num_batch  # (N, None) int32, (N, None) int32, ()
