
import tensorflow as tf

hparams = tf.contrib.training.HParams(

    graphemes_num=29,
    phonemes_num=73,

    max_grapheme_seq_len=35,
    max_phoneme_seq_len=35,

    hidden_units=256,
    dropout_rate=0.5,
    encoder_blocks=2,
    decoder_blocks=2,
    heads_num=2,

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
