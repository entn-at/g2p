
import tensorflow as tf

hparams = tf.contrib.training.HParams(

    graphemes_num=28,
    phonemes_num=72,
    nbest=5,
    embedding_dim=512,
    zonout_prob=(0.1, 0.01),
    dropout_rate=0.5,
    with_conv=True,
    conv_num=3,
    conv_width=5,
    conv_channels=512,
    bilstm_layers_num=1,
    rnn_type='zn_lstm',
    lstm_units1=512,
    lstm_units2=128,

    batch_size=256,
    group_size=4,
    lr=0.001,
    lr_hl=20000,
    grad_clip_ratio=1.0,
    max_steps=50000,
    save_every_nth=1000,
    eval_every_nth=1000,
    reshuffle_every_nth=5,
)
