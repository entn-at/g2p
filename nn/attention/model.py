
import tensorflow as tf
from tensorflow.contrib.rnn import OutputProjectionWrapper
from tensorflow.contrib.seq2seq import BahdanauAttention, AttentionWrapper
from tensorflow.python.layers import core as layers_core

from nn.modules import *


class G2PModel:

    @staticmethod
    def create_attention_cell(depth, memory, seq_len, cell,
                              alignment_history=False):
        attention = BahdanauAttention(depth,
                                      memory,
                                      memory_sequence_length=seq_len,
                                      normalize=True)
        attention_cell = AttentionWrapper(cell,
                                          attention,
                                          alignment_history=alignment_history)
        return attention_cell

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
            outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(
                forward_cell, backward_cell,
                outputs, sequence_length=self.input_lengths, dtype=tf.float32,
                scope='bilstm')

            # Concatentate forward and backwards:
            encoder_outputs = tf.concat(outputs, axis=2)

            decoder_cell = MultiRNNCell([rnn_cell(hparams.decoder_lstm_units, hparams, is_training),
                                         rnn_cell(hparams.decoder_lstm_units, hparams, is_training)],
                                         state_is_tuple=True)
            decoder_embeddings = tf.get_variable(name='decoder_embeddings',
                                                 shape=[hparams.phonemes_num,
                                                        hparams.decoder_embedding_dim],
                                                 dtype=tf.float32)

            if is_training:
                batch_size = tf.shape(self.inputs)[0]
                attention_cell = self.create_attention_cell(
                        hparams.attention_depth, encoder_outputs,
                        self.input_lengths, decoder_cell,
                        alignment_history=False)
                attention_cell = OutputProjectionWrapper(attention_cell, hparams.phonemes_num)
                targets_shifted = self.targets[:, :-1]
                targets_emb = tf.nn.embedding_lookup(decoder_embeddings,
                                                     targets_shifted)
                helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs=targets_emb, sequence_length=self.target_lengths)
                #decoder_initial_state = attention_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)
                decoder_initial_state = attention_cell.zero_state(batch_size, tf.float32)
                decoder = tf.contrib.seq2seq.BasicDecoder(attention_cell, helper,
                        decoder_initial_state)
                outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
                self.decoded_best = tf.identity(outputs.sample_id, name='predicted_1best')
                self.logits = outputs.rnn_output
                self.probs = tf.nn.softmax(self.logits, name='probs')
            else:
                if self.hparams.beam_width == 1:
                    batch_size = tf.shape(self.inputs)[0]
                    attention_cell = self.create_attention_cell(
                            hparams.attention_depth, encoder_outputs,
                            self.input_lengths, decoder_cell,
                            alignment_history=False)
                    attention_cell = OutputProjectionWrapper(attention_cell, hparams.phonemes_num)
                    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                            embedding=decoder_embeddings,
                            start_tokens=tf.fill([batch_size], hparams.phonemes_num-2),
                            end_token=hparams.phonemes_num-1)
                    decoder_initial_state = attention_cell.zero_state(batch_size, tf.float32)
                    decoder = tf.contrib.seq2seq.BasicDecoder(attention_cell, helper,
                            decoder_initial_state)
                    outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                            maximum_iterations=self.hparams.max_phoneme_seq_len)
                    self.decoded_best = tf.identity(outputs.sample_id, name='predicted_1best')
                    self.logits = outputs.rnn_output
                    self.probs = tf.nn.softmax(self.logits, name='probs')
                else:
                    batch_size = tf.shape(self.inputs)[0]
                    start_tokens = tf.fill([batch_size], hparams.phonemes_num-2)
                    batch_size = batch_size * hparams.beam_width
                    encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=hparams.beam_width)
                    input_lengths_tile = tf.contrib.seq2seq.tile_batch(self.input_lengths, multiplier=hparams.beam_width)
                    encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=hparams.beam_width)

                    attention_cell = self.create_attention_cell(
                            hparams.attention_depth, encoder_outputs,
                            input_lengths_tile, decoder_cell,
                            alignment_history=False)
                    attention_cell = OutputProjectionWrapper(attention_cell, hparams.phonemes_num)
                    decoder_initial_state = attention_cell.zero_state(batch_size, tf.float32)
                    decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell=attention_cell, embedding=decoder_embeddings,
                        start_tokens=start_tokens, end_token=hparams.phonemes_num-1,
                        initial_state=decoder_initial_state,
                        beam_width=hparams.beam_width,
                        output_layer=None,
                        length_penalty_weight=hparams.length_penalty)
                    outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                        maximum_iterations=hparams.max_iters)
                    self.logits = tf.no_op()
                    print('**Warning! You could not be able to build lattice with beam_width > 1')
                    self.probs = tf.no_op()
                    # best beam
                    self.decoded_best = tf.identity(outputs.predicted_ids[:, :, 0],
                                                    name='predicted_1best')
                    #self.alignment = tf.transpose(final_state.alignment_history.stack(), [1, 2, 0])

    def add_loss(self):
        assert self.with_target
        targets_shifted = self.targets[:, 1:]
        max_target_len = tf.shape(targets_shifted)[1]
        batch_size = tf.shape(targets_shifted)[0]
        mask = tf.sequence_mask(self.target_lengths, max_target_len,
                                dtype=tf.float32)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targets_shifted, logits=self.logits)
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
