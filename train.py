# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
import tensorflow as tf

from hyperparams import Hyperparams as hp
from modules import *
import os, codecs
from tqdm import tqdm
from data_load import load_embeddings, load_vocab, load_dataset, load_test_dataset
from model import mk_model
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

total_samples = 128000 * 50
num_gpu = 3
batch_size = 256
num_batch = total_samples // batch_size

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = [g for g, _ in grad_and_vars]
        # Average over the 'tower' dimension.
        grad = tf.stack(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads

class Graph():
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            embeddings = load_embeddings()
            eos_i = len(embeddings) - 1
            sos_i = eos_i - 1

            if is_training:
               self.training_dataset = load_dataset(batch_size, sos_i, eos_i)
               training_iterator = self.training_dataset.make_one_shot_iterator()
               self.iterator = training_iterator

            self.embeddings = np.array(embeddings)

            with tf.device('/cpu:0'):
                if is_training:
                    self.models = []
                    self.learning_rate = tf.placeholder(tf.float32, shape=[])
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-8)

                    self.global_step = tf.Variable(0, name='global_step', trainable=False)

                    self.build_gpu_models(is_training)

                    tower_preds, tower_losses, tower_grads, tower_hits, tower_istarget = zip(*self.models)
                    self.aver_loss_op = tf.reduce_mean(tower_losses)
                    self.apply_gradient_op = self.optimizer.apply_gradients(average_gradients(tower_grads), global_step=self.global_step)

                    self.accuracy = tf.reduce_sum(tower_hits) / tf.reduce_sum(tower_istarget)

                    tf.summary.scalar('mean_loss', self.aver_loss_op)
                    tf.summary.scalar('acc', self.accuracy)
                    self.merge = tf.summary.merge_all()

    def build_1_gpu(self, gpu_id, is_training):
        with tf.name_scope('tower_%d' % gpu_id):
            with tf.variable_scope('cpu_variables', reuse=gpu_id>0):
                X, Y, YL, SL = self.iterator.get_next()

                outputs, preds = mk_model(X, Y, self.embeddings, is_training)

                y_smoothed = label_smoothing(tf.one_hot(YL, depth=len(embeddings)))

                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=YL, logits=outputs)

                istarget = tf.to_float(tf.not_equal(YL, tf.zeros_like(Y)))  # masking

                hits = tf.to_float(tf.equal(preds, YL)) * istarget

                sum_hits = tf.reduce_sum(hits)
                sum_istarget = tf.reduce_sum(istarget)

                mean_loss = tf.reduce_sum(crossent * istarget) / sum_istarget
                grads = self.optimizer.compute_gradients(mean_loss)
                self.models.append((preds, mean_loss, grads, sum_hits, sum_istarget))

    def build_one_gpu_model(self, gpu_id, is_training):
        with tf.device('/gpu:%d' % gpu_id):
            print('tower:%d...'% gpu_id)
            self.build_1_gpu(gpu_id, is_training)

    def build_gpu_models(self, is_training):
        for gpu_id in range(num_gpu):
            self.build_one_gpu_model(gpu_id, is_training)


if __name__ == '__main__':
    # Construct graph
    g = Graph(); print("Graph loaded")

    with g.graph.as_default(), tf.device("/cpu:0"):
        sv = tf.train.Supervisor(logdir=hp.logdir,
                                 save_model_secs=0,
                                 summary_op=None)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95

        with sv.managed_session(config=config) as sess:
            lr = hp.lr
            for epoch in range(1, hp.num_epochs + 1):
                if sv.should_stop():
                     break

                avg_loss = 0.0
                avg_acc = 0.0

                n_steps = num_batch // num_gpu
                for step in tqdm(range(n_steps), total=n_steps, ncols=70, leave=False, unit='b'):
                    inp_dict = { }
                    inp_dict[g.learning_rate] = lr
                    _, _loss, _acc = sess.run([g.apply_gradient_op, g.aver_loss_op, g.accuracy], inp_dict)

                    if step % 100 == 0:
                      sv.summary_computed(sess, sess.run(g.merge))

                    avg_loss += _loss
                    avg_acc  += _acc

                avg_loss /= n_steps
                avg_acc /=  n_steps

                print('epoch: %03d, train loss:%.4f, acc: %.4f, lr: %.5f' % (epoch, avg_loss, avg_acc, lr))
                if epoch % 11 == 0:
                   lr = max(lr * 0.7,0.00001)

    print("Done")
