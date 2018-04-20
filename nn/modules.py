
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, LSTMBlockCell, \
    LayerNormBasicLSTMCell, MultiRNNCell, OutputProjectionWrapper

class ZoneoutWrapper(RNNCell):
    """Wrapper for the TF RNN cell
    Operator adding zoneout to all states (states+cells) of the given cell.
    For an LSTM, the 'cell' is a tuple containing state and cell

    Usage:
    cell = ZoneoutWrapper(tf.nn.rnn_cell.LSTMCell(256), zoneout_prob=(0.05, 0))
    cell = ZoneoutWrapper(tf.nn.rnn_cell.GRUCell(256), zoneout_prob=0.05)
    """
    def __init__(self, cell, zoneout_prob, is_training=True, seed=None):
        super(ZoneoutWrapper, self).__init__()
        if not isinstance(cell, tf.nn.rnn_cell.RNNCell):
            raise TypeError("The parameter cell is not an RNNCell.")
        if (isinstance(zoneout_prob, float) and 
                not (zoneout_prob >= 0.0 and zoneout_prob <= 1.0)):
            raise ValueError("Param zoneout_prob must be between 0 and 1: %d"
                       % zoneout_prob)
        self._cell = cell
        self._zoneout_prob = tuple(zoneout_prob)
        self._seed = seed
        self.is_training = is_training
  
    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        if isinstance(self.state_size, tuple) \
                != isinstance(self._zoneout_prob, tuple):
            raise TypeError("Subdivided states need subdivided zoneouts.")
        if (isinstance(self.state_size, tuple) and 
                len(tuple(self.state_size)) != len(tuple(self._zoneout_prob))):
            raise ValueError("State and zoneout need equally many parts.")
        output, new_state = self._cell(inputs, state, scope)
        if isinstance(self.state_size, tuple):
            if self.is_training:
                (c, h) = tuple((1 - state_part_zoneout_prob) * tf.nn.dropout(
                                new_state_part - state_part, 
                                (1 - state_part_zoneout_prob), 
                                seed=self._seed) + state_part
                            for new_state_part, 
                                state_part, 
                                state_part_zoneout_prob 
                                in zip(new_state, state, self._zoneout_prob))
            else:
                (c, h) = tuple(state_part_zoneout_prob * state_part 
                                + (1 - state_part_zoneout_prob) 
                                * new_state_part
                            for new_state_part, 
                                state_part, 
                                state_part_zoneout_prob 
                                in zip(new_state, state, self._zoneout_prob))
            new_state = tf.nn.rnn_cell.LSTMStateTuple(c, h)
        else:
            if self.is_training:
                new_state = (1 - self._zoneout_prob) * tf.nn.dropout(
                                new_state - state, (1 - self._zoneout_prob), 
                                    seed=self._seed) + state
            else:
                new_state = self._zoneout_prob * state \
                                + (1 - self._zoneout_prob) \
                                * new_state

        return output, new_state

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)


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


def rnn_cell(dim, hparams, is_training):
    if hparams.rnn_type == 'ln_lstm':
        keep_prob = (1 - hparams.dropout_rate) if is_training else 1.0
        cell = LayerNormBasicLSTMCell(dim, dropout_keep_prob=keep_prob,
                                      layer_norm=True)
    elif hparams.rnn_type == 'zn_lstm':
        cell = LSTMBlockCell(dim)
        cell = ZoneoutWrapper(cell, hparams.zonout_prob, is_training)
    return cell
