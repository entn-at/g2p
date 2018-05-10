
import tensorflow as tf

from nn.modules import *
from nn.transformer.specific_modules import *


def encoder(inputs, hparams, is_training=False, reuse=False):
    with tf.variable_scope('encoder', reuse=reuse):
        ## Embedding
        enc = embedding(inputs, vocab_size=hparams.graphemes_num,
                        num_units=hparams.hidden_units,
                        scale=True,
                        scope="enc_embed")
        if hparams.positional_encoding:
            enc += positional_encoding(inputs, 
                                       vocab_size=hparams.max_grapheme_seq_len,
                                       num_units=hparams.hidden_units,
                                       scale=False,
                                       scope="enc_pe")

        enc = tf.layers.dropout(enc, rate=hparams.dropout_rate,
                                training=tf.convert_to_tensor(is_training))

        for i in range(hparams.encoder_blocks):
            with tf.variable_scope("num_blocks_{}".format(i)):
                ### Multihead Attention
                enc = multihead_attention(queries=enc, keys=enc,
                                          num_units=hparams.hidden_units,
                                          num_heads=hparams.heads_num,
                                          dropout_rate=hparams.dropout_rate,
                                          is_training=is_training,
                                          causality=False, reuse=reuse)

                ### Feed Forward
                enc = feedforward(enc, num_units=[4 * hparams.hidden_units,
                                  hparams.hidden_units], reuse=reuse)
        return enc


def decoder(enc, decoder_inputs, hparams, is_training=False, reuse=False,
            embed_input=True):
    with tf.variable_scope('decoder', reuse=reuse):
        if embed_input:
            ## Embedding
            dec = embedding(decoder_inputs,
                            vocab_size=hparams.phonemes_num,
                            num_units=hparams.hidden_units,
                            scale=True,
                            scope="dec_embed")
        else:
            dec = decoder_inputs

        if hparams.positional_encoding:
            dec += positional_encoding(decoder_inputs,
                                       vocab_size=hparams.max_phoneme_seq_len,
                                       num_units=hparams.hidden_units,
                                       scale=False,
                                       scope="dec_pe")

        ## Dropout
        dec = tf.layers.dropout(dec, rate=hparams.dropout_rate,
                                training=tf.convert_to_tensor(is_training))

        for i in range(hparams.decoder_blocks):
            with tf.variable_scope("num_blocks_{}".format(i)):
                ## Multihead Attention ( self-attention)
                dec = multihead_attention(queries=dec,
                                          keys=dec,
                                          num_units=hparams.hidden_units,
                                          num_heads=hparams.heads_num,
                                          dropout_rate=hparams.dropout_rate,
                                          is_training=is_training,
                                          causality=True,
                                          scope="self_attention", reuse=reuse)

                ## Multihead Attention ( vanilla attention)
                dec = multihead_attention(queries=dec,
                                          keys=enc,
                                          num_units=hparams.hidden_units,
                                          num_heads=hparams.heads_num,
                                          dropout_rate=hparams.dropout_rate,
                                          is_training=is_training,
                                          causality=False,
                                          scope="vanilla_attention", reuse=reuse)

                ## Feed Forward
                dec = feedforward(dec, num_units=[4 * hparams.hidden_units,
                                                  hparams.hidden_units], reuse=reuse)

        # Project
        dec = tf.layers.dense(dec, hparams.phonemes_num)
        return dec


class G2PModel:
    def __init__(self, hparams, is_training=False, with_target=True, reuse=False):
        self.with_target = with_target
        self.is_training = is_training
        self.hparams = hparams
        self.inputs = tf.placeholder(tf.int32, (None, None), name='graphemes_ph')
        self.input_lengths = tf.placeholder(tf.int32, [None], name='grapeheme_seq_len_ph')
        if with_target:
            self.targets = tf.placeholder(tf.int32, (None, None), name='phonemes_ph')
            self.target_lengths = tf.placeholder(tf.int32, [None], name='phoneme_seq_len_ph')

        with tf.variable_scope('g2p', reuse=reuse):
            self.enc = encoder(self.inputs, self.hparams, is_training=is_training, reuse=reuse)

            if is_training:
                decoder_inputs = self.targets[:, :-1]
                self.logits = decoder(self.enc, decoder_inputs, self.hparams,
                                      is_training=is_training, reuse=reuse)
                self.decoded_best = tf.to_int32(tf.arg_max(self.logits, dimension=-1))
            else:
                batch_size = tf.shape(self.inputs)[0]
                arr = np.zeros((1, 1, self.hparams.phonemes_num))
                arr[0, 0, self.hparams.phonemes_num - 2] = 1
                arr = tf.convert_to_tensor(arr, dtype=tf.float32)
                decoder_inputs0 = tf.tile(arr, [batch_size, 1, 1])
                i_0 = tf.constant(0)

                condition = lambda i, inpts: i <  tf.reduce_max(self.input_lengths) + 3

                def body(i, inpts):
                    otpts = decoder(self.enc, tf.to_int32(tf.arg_max(inpts, dimension=-1)),
                                    self.hparams, is_training=False, reuse=reuse)
                    inpts = tf.concat([inpts, otpts[:, -1:, :]], 1)
                    return i+1, inpts

                _, dec = tf.while_loop(condition, body, loop_vars=[i_0, decoder_inputs0],
                        shape_invariants=[i_0.get_shape(), tf.TensorShape([None, None, self.hparams.phonemes_num])])
                self.decoded_best = tf.identity(tf.to_int32(tf.arg_max(dec[:, 1:], dimension=-1)),
                                                name='predicted_1best')
                self.probs = tf.nn.softmax(dec, name='probs')

    def add_loss(self):
        decoder_targets = self.targets[:, 1:]
        if self.hparams.label_smoothing:
            decoder_targets_sm = label_smoothing(tf.one_hot(decoder_targets,
                                                            self.hparams.phonemes_num))
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                           labels=decoder_targets_sm)
        else:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=decoder_targets, logits=self.logits)
        max_target_len = tf.shape(decoder_targets)[1]
        mask = tf.sequence_mask(self.target_lengths, max_target_len,
                                                dtype=tf.float32)
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
