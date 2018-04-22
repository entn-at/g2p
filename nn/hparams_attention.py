
import tensorflow as tf

hparams = tf.contrib.training.HParams(

    graphemes_num=29,
    phonemes_num=73,

    embedding_dim=512,
    zonout_prob=(0.1, 0.01),
    dropout_rate=0.5,
    with_conv=True,
    conv_num=3,
    conv_width=5,
    conv_channels=512,

    rnn_type='zn_lstm',
    encoder_lstm_units=512,
    attention_depth=256,
    beam_width=10,
    length_penalty=0.0,
    decoder_lstm_units=512,
    decoder_embedding_dim=512,
    max_iters=50,

    batch_size=256,
    group_size=16,
    lr=0.001,
    lr_hl=100000,
    grad_clip_ratio=1.0,
    max_steps=500000,
    save_every_nth=1000,
    eval_every_nth=2000,
    reshuffle_every_nth=5,
)
