
import tensorflow as tf
from tensorflow.contrib.rnn import OutputProjectionWrapper
from tensorflow.contrib.seq2seq import BahdanauAttention, AttentionWrapper

from nn.modules import *


class G2PModel:

    def __init__(self, hparams, is_training=False, with_target=True, reuse=False):
        self.with_target = with_target
        self.hparams = hparams
        self.is_training = is_training
        self.inputs = tf.placeholder(tf.int32, (None, None),
                                     name='graphemes_ph')
        self.input_lengths = tf.placeholder(tf.int32, [None],
                                            name='grapeheme_seq_len_ph')
        if with_target:
            self.targets = tf.placeholder(tf.int32, (None, None), name='phonemes_ph')
            self.target_lengths = tf.placeholder(tf.int32, [None], name='phoneme_seq_len_ph')

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

            forward_cell = rnn_cell(hparams.encoder_lstm_units//2, hparams, is_training)
            backward_cell = rnn_cell(hparams.encoder_lstm_units//2, hparams, is_training)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                forward_cell, backward_cell,
                outputs, sequence_length=self.input_lengths, dtype=tf.float32,
                scope='bilstm')

            # Concatentate forward and backwards:
            encoder_outputs = tf.concat(outputs, axis=2)

            decoder_cell = rnn_cell(hparams.decoder_lstm_units, hparams, is_training)
            decoder_embeddings = tf.get_variable(name='decoder_embeddings',
                                                 shape=[hparams.phonemes_num,
                                                        hparams.decoder_embedding_dim],
                                                 dtype=tf.float32)
            batch_size = tf.shape(self.inputs)[0]

            attention = BahdanauAttention(hparams.attention_depth,
                                          encoder_outputs,
                                          memory_sequence_length=self.input_lengths)
            attention_cell = AttentionWrapper(decoder_cell,
                                              attention,
                                              alignment_history=True)
            attention_cell_proj = OutputProjectionWrapper(attention_cell,
                                                          hparams.phonemes_num)

            if is_training:
                targets_shifted = self.targets[:, :-1]
                targets_emb = tf.nn.embedding_lookup(decoder_embeddings,
                                                     targets_shifted)
                helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs=targets_emb, sequence_length=self.target_lengths)
            else:
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding=decoder_embeddings,
                    start_tokens=tf.fill([batch_size], hparams.phonemes_num-2),
                    end_token=hparams.phonemes_num-1)

            # TODO figure out BeamSearchDecoder (problems with computing the loss)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=attention_cell_proj,
                helper=helper,
                initial_state=attention_cell_proj.zero_state(batch_size,
                                                             dtype=tf.float32))

            (self.logits, _), final_state, self.decoder_length = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder,
                impute_finished=False,
                maximum_iterations=tf.reduce_max(self.target_lengths) if with_target else hparams.max_iters)
            self.decoded_best = tf.argmax(self.logits, axis=-1)
            self.alignment = tf.transpose(final_state.alignment_history.stack(), [1, 2, 0])

    def add_loss(self):
        assert self.with_target
        targets_shifted = self.targets[:, 1:]

        max_target_len = tf.shape(targets_shifted)[1]
        if not self.is_training:

            logits_pad = tf.pad(self.logits,
                                [[0, 0], [0, max_target_len - tf.shape(self.logits)[1]], [0, 0]],
                                constant_values=self.hparams.phonemes_num-1)
        else:
            logits_pad = self.logits

        mask = tf.sequence_mask(self.target_lengths, max_target_len,
                                dtype=tf.float32)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targets_shifted, logits=logits_pad)
        self.loss = tf.reduce_mean(loss * mask)


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
            fd[self.target_lengths] = batch[3]
        return fd