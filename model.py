
import tensorflow as tf
from modules import *
from hyperparams import Hyperparams as hp

def mk_encoder(x, embedding, is_training):
    enc = pretrain_embed(x, embedding, trainable=True, scale=True)

    with tf.variable_scope("encoder"):
        ## Positional Encoding
        enc += positional_encoding(x,
                              num_units=hp.hidden_units,
                              zero_pad=False,
                              scale=False,
                              scope="enc_pe")


        ## Dropout
        enc = tf.layers.dropout(enc,
                                rate=hp.dropout_rate,
                                training=tf.convert_to_tensor(is_training))

        ## Blocks
        for i in range(hp.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i)):
                ### Multihead Attention
                enc = multihead_attention(queries=enc,
                                          keys=enc,
                                          num_units=hp.hidden_units,
                                          num_heads=hp.num_heads,
                                          dropout_rate=hp.dropout_rate,
                                          is_training=is_training,
                                          causality=False)

                ### Feed Forward
                enc = feedforward(enc, num_units=[4*hp.hidden_units, hp.hidden_units])

        return enc


def mk_decoder(y, enc, embedding, is_training):
    dec = pretrain_embed(y, embedding, trainable=True, scale=True)

    with tf.variable_scope("decoder"):
        ## Positional Encoding
        dec += positional_encoding(y,
                              vocab_size=hp.maxlen,
                              num_units=hp.hidden_units,
                              zero_pad=False,
                              scale=False,
                              scope="dec_pe")

        ## Dropout
        dec = tf.layers.dropout(dec,
                                rate=hp.dropout_rate,
                                training=tf.convert_to_tensor(is_training))

        ## Blocks
        for i in range(hp.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i)):
                ## Multihead Attention ( self-attention)
                dec = multihead_attention(queries=dec,
                                          keys=dec,
                                          num_units=hp.hidden_units,
                                          num_heads=hp.num_heads,
                                          dropout_rate=hp.dropout_rate,
                                          is_training=is_training,
                                          causality=True,
                                          scope="self_attention")

                ## Multihead Attention ( vanilla attention)
                dec = multihead_attention(queries=dec,
                                          keys=enc,
                                          num_units=hp.hidden_units,
                                          num_heads=hp.num_heads,
                                          dropout_rate=hp.dropout_rate,
                                          is_training=is_training,
                                          causality=False,
                                          scope="vanilla_attention")

                ## Feed Forward
                dec = feedforward(dec, num_units=[4*hp.hidden_units, hp.hidden_units])



        return enc

def mk_model(x, y, embedding, is_training):
   enc = mk_encoder(x, embedding, is_training)
   dec = mk_decoder(y, enc, embedding, is_training)

   shape = embedding.shape

   logits = tf.layers.dense(dec, shape[0])
   preds = tf.to_int32(tf.arg_max(logits, dimension=-1))

   return logits, preds
