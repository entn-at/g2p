
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMBlockCell

from nn.modules import *


class G2PModel:
    def __init__(self, hparams, is_training=False, with_target=True, reuse=False):
        self.with_target = with_target
        self.hparams = hparams
        self.inputs = tf.placeholder(tf.int32, (None, None), name='graphemes_ph')
        self.input_lengths = tf.placeholder(tf.int32, [None], name='grapeheme_seq_len_ph')
        if with_target:
            self.targets = tf.sparse_placeholder(tf.int32, name='phonemes_ph')

        with tf.variable_scope('g2p', reuse=reuse):
            embedding_table = tf.get_variable('embedding',
                                              [hparams.graphemes_num, hparams.embedding_dim],
                                              dtype=tf.float32,
                                              initializer=tf.truncated_normal_initializer(stddev=0.5))
            outputs = tf.nn.embedding_lookup(embedding_table, self.inputs)

            if hparams.with_conv:
                for i in range(hparams.conv_num):
                    outputs = conv1d(outputs, hparams.conv_width,
                                     hparams.conv_channels, tf.nn.relu,
                                     is_training, hparams.dropout_rate,
                                     'conv_%d' % i)

            forward_cell = rnn_cell(hparams.lstm_units1//2, hparams, is_training)
            backward_cell = rnn_cell(hparams.lstm_units1//2, hparams, is_training)
            bi_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                forward_cell, backward_cell,
                outputs, sequence_length=self.input_lengths, dtype=tf.float32,
                scope='bilstm')

            # Concatentate forward and backwards:
            bi_outputs = tf.concat(bi_outputs, axis=2)

            uni_cell = rnn_cell(hparams.lstm_units1, hparams, is_training)

            uni_outputs, _ = tf.nn.dynamic_rnn(uni_cell, outputs,
                                               sequence_length=self.input_lengths,
                                               dtype=tf.float32,
                                               scope='unilstm')
            outputs = tf.concat([bi_outputs, uni_outputs], axis=2)

            dropout_rate_cond = hparams.dropout_rate if is_training else 0.0
            outputs = tf.layers.dropout(outputs, rate=dropout_rate_cond)

            outputs, _ = tf.nn.dynamic_rnn(LSTMBlockCell(hparams.lstm_units2),
                                           outputs,
                                           sequence_length=self.input_lengths,
                                           dtype=tf.float32,
                                           scope='lstm')
            self.logits = tf.layers.dense(outputs, hparams.phonemes_num)
            self.probs = tf.nn.softmax(self.logits, name='probs')

            logits_transp = tf.transpose(self.logits, (1, 0, 2))
            self.decoded, self.seq_probs = tf.nn.ctc_beam_search_decoder(
                logits_transp, self.input_lengths, top_paths=self.hparams.nbest)
            self.decoded_best = tf.sparse_tensor_to_dense(self.decoded[0],
                                                          name='predicted_1best')

    def add_loss(self):
        assert self.with_target
        loss = tf.nn.ctc_loss(self.targets, self.logits, self.input_lengths,
                              time_major=False)
        self.loss = tf.reduce_mean(loss)

    def add_train_and_stats(self):
        with tf.variable_scope('training'):
            self.global_step = tf.Variable(0, name='global_step',
                                           trainable=False)
            lr = tf.train.exponential_decay(self.hparams.lr,
                                            self.global_step,
                                            self.hparams.lr_hl, 0.5)
            opt = tf.train.AdamOptimizer(learning_rate=lr)
            assert(hasattr(self, 'loss'))
            gradients, variables = zip(*opt.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients,
                                                  self.hparams.grad_clip_ratio)
            with tf.control_dependencies(
                    tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.train_op = opt.apply_gradients(zip(gradients, variables),
                                                    global_step=self.global_step)

        with tf.variable_scope('stats'):
            tf.summary.histogram('phoneme_probs', self.logits)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('learning_rate', lr)
            gradient_norms = [tf.norm(grad) for grad in gradients]
            tf.summary.histogram('gradient_norm', gradient_norms)
            tf.summary.scalar('max_gradient_norm',
                              tf.reduce_max(gradient_norms))
            self.stats_op = tf.summary.merge_all()

    def create_feed_dict(self, batch):
        fd = {self.inputs: batch[0], self.input_lengths: batch[1]}
        if self.with_target:
            fd[self.targets] = batch[2]
        return fd
