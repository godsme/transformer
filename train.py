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
from data_load import get_batch_data, load_de_vocab, load_en_vocab
from modules import *
import os, codecs
from tqdm import tqdm

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
    g = Graph("train"); print("Graph loaded")

    # Start session
    sv = tf.train.Supervisor(graph=g.graph,
                             logdir=hp.logdir,
                             save_model_secs=0)
    with sv.managed_session() as sess:
        for epoch in range(1, hp.num_epochs+1):
            if sv.should_stop(): break
            for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                sess.run(g.train_op)

            gs = sess.run(g.global_step)
            sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))

    print("Done")
