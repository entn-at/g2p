
import tensorflow as tf

from nn.modules import *
from nn.transformer.specific_modules import *


class G2PModel:
    def __init__(self, hparams, is_training=False, with_target=True, reuse=False):
        self.with_target = with_target
        self.hparams = hparams
        self.inputs = tf.placeholder(tf.int32, (None, None), name='graphemes_ph')
        self.input_lengths = tf.placeholder(tf.int32, [None], name='grapeheme_seq_len_ph')
        if with_target:
            self.targets = tf.placeholder(tf.int32, (None, None), name='phonemes_ph')
            self.target_lengths = tf.placeholder(tf.int32, [None], name='phoneme_seq_len_ph')

        with tf.variable_scope('g2p', reuse=reuse):
            with tf.variable_scope("encoder"):
                ## Embedding
                self.enc = embedding(self.inputs,
                                      vocab_size=self.hparams.graphemes_num,
                                      num_units=self.hparams.hidden_units,
                                      scale=True,
                                      scope="enc_embed")
                self.enc += positional_encoding(self.inputs,
                                                vocab_size=self.hparams.max_grapheme_seq_len,
                                                num_units=self.hparams.hidden_units,
                                                scale=False,
                                                scope="enc_pe")

                self.enc = tf.layers.dropout(self.enc,
                                             rate=hparams.dropout_rate,
                                             training=tf.convert_to_tensor(
                                                 is_training))

            for i in range(self.hparams.encoder_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    self.enc = multihead_attention(queries=self.enc,
                                                   keys=self.enc,
                                                   num_units=self.hparams.hidden_units,
                                                   num_heads=self.hparams.heads_num,
                                                   dropout_rate=self.hparams.dropout_rate,
                                                   is_training=is_training,
                                                   causality=False)

                    ### Feed Forward
                    self.enc = feedforward(self.enc,
                                           num_units=[4 * self.hparams.hidden_units,
                                                      self.hparams.hidden_units])

        with tf.variable_scope("decoder"):
            if is_training:
                decoder_inputs = self.targets[:, :-1]

                ## Embedding
                self.dec = embedding(decoder_inputs,
                                      vocab_size=self.hparams.graphemes_num,
                                      num_units=self.hparams.hidden_units,
                                      scale=True,
                                      scope="dec_embed")

                self.dec += positional_encoding(decoder_inputs,
                                                vocab_size=self.hparams.max_phoneme_seq_len,
                                                num_units=self.hparams.hidden_units,
                                                scale=False,
                                                scope="dec_pe")

                ## Dropout
                self.dec = tf.layers.dropout(self.dec,
                                            rate=self.hparams.dropout_rate,
                                            training=tf.convert_to_tensor(is_training))

                for i in range(self.params.decoder_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ## Multihead Attention ( self-attention)
                        self.dec = multihead_attention(queries=self.dec,
                                                       keys=self.dec,
                                                       num_units=self.hparams.hidden_units,
                                                       num_heads=self.hparams.heads_num,
                                                       dropout_rate=self.hparams.dropout_rate,
                                                       is_training=is_training,
                                                       causality=True,
                                                       scope="self_attention")

                        ## Multihead Attention ( vanilla attention)
                        self.dec = multihead_attention(queries=self.dec,
                                                       keys=self.enc,
                                                       num_units=self.hparams.hidden_units,
                                                       num_heads=self.hparams.num_heads,
                                                       dropout_rate=self.hparams.dropout_rate,
                                                       is_training=is_training,
                                                       causality=False,
                                                       scope="vanilla_attention")

                        ## Feed Forward
                        self.dec = feedforward(self.dec,
                                               num_units=[4 * self.hparams.hidden_units,
                                                          self.hparams.hidden_units])

        # Final linear projection
        self.logits = tf.layers.dense(self.dec, self.hparams.phonemes_num)
        self.decoded_best = tf.to_int32(tf.arg_max(self.logits, dimension=-1))

    def add_loss(self):
        decoder_targets = self.targets[:, 1:]
        decoder_targets_sm = label_smoothing(tf.one_hot(self.decoder_targets,
                                                        self.hparams.phonemes_num))
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                            labels=self.decoder_targets_sm)

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
