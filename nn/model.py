
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMBlockCell, LayerNormBasicLSTMCell


def conv1d(inputs, kernel_size, channels, activation, is_training, drop_rate, scope):
    with tf.variable_scope(scope):
        conv1d_output = tf.layers.conv1d(
                inputs,
                filters=channels,
                kernel_size=kernel_size,
                activation=activation,
                padding='same')
        drop_rate_cond = drop_rate if is_training else 0.0
        conv1d_output = tf.layers.dropout(conv1d_output, rate=drop_rate_cond)
    return tf.layers.batch_normalization(conv1d_output, training=is_training)


def create_placeholders(with_target=False):
    placeholders = [tf.placeholder(tf.int32, (None, None), name='graphemes_ph')]
    input_lengths = tf.placeholder(tf.int32, None, name='grapeheme_seq_len_ph')
    placeholders.append(input_lengths)
    if with_target:
        targets = tf.sparse_placeholder(tf.int32, name='phonemes_ph')
        placeholders.append(targets)
    return placeholders


def rnn_cell(dim, is_training):
    keep_prob = 0.5 if is_training else 1.0
    cell = LayerNormBasicLSTMCell(dim, dropout_keep_prob=keep_prob, layer_norm=True)
    return cell


class G2PModel:
    def __init__(self, inputs, input_lengths, hparams, is_training=False,
                 reuse=False):
        with tf.variable_scope('g2p', reuse=reuse):
            self.input_lengths = input_lengths
            self.hparams = hparams
            embedding_table = tf.get_variable('embedding',
                                              [hparams.graphemes_num, hparams.embedding_dim],
                                              dtype=tf.float32,
                                              initializer=tf.truncated_normal_initializer(stddev=0.5))
            outputs = tf.nn.embedding_lookup(embedding_table, inputs)

            if hparams.with_conv:
                for i in range(hparams.conv_num):
                    outputs = conv1d(outputs, hparams.conv_width,
                                     hparams.conv_channels, tf.nn.relu,
                                     is_training, hparams.dropout_rate,
                                     'conv_%d' % i)

            forward_cell = rnn_cell(hparams.lstm_units1//2, is_training)
            backward_cell = rnn_cell(hparams.lstm_units1//2, is_training)
            bi_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                forward_cell, backward_cell,
                outputs, sequence_length=input_lengths, dtype=tf.float32,
                scope='bilstm')

            # Concatentate forward and backwards:
            bi_outputs = tf.concat(bi_outputs, axis=2)

            uni_cell = rnn_cell(hparams.lstm_units1, is_training)

            uni_outputs, _ = tf.nn.dynamic_rnn(uni_cell, outputs,
                                               sequence_length=input_lengths,
                                               dtype=tf.float32,
                                               scope='unilstm')
            outputs = tf.concat([bi_outputs, uni_outputs], axis=2)

            outputs, _ = tf.nn.dynamic_rnn(LSTMBlockCell(hparams.lstm_units2),
                                           outputs,
                                           sequence_length=input_lengths,
                                           dtype=tf.float32,
                                           scope='lstm')
            self.logits = tf.layers.dense(outputs, hparams.phonemes_num)

    def add_loss(self, targets):
        loss = tf.nn.ctc_loss(targets, self.logits, self.input_lengths,
                              time_major=False)
        self.loss = tf.reduce_mean(loss)

    def add_decoder(self):
        logits = tf.transpose(self.logits, (1, 0, 2))
        self.decoded, self.probs = tf.nn.ctc_beam_search_decoder(
            logits, self.input_lengths, top_paths=self.hparams.nbest)

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