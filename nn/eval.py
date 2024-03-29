
import os
import sys
import time
import json
import string
import argparse

import tensorflow as tf

from nn.encode_utils import *
from nn.utils import *
from read_utils import *


def parse_args():
    arg_parser = argparse.ArgumentParser(description='Evals G2P model')
    arg_parser.add_argument('--dict', required=True,
                            help='Dictionary to train on')
    arg_parser.add_argument('--model-dir', default='g2p_model', required=True,
                            help='Directory to put model to')
    arg_parser.add_argument('--restore', type=int, required=True,
                            help='Step to restore from')

    args = arg_parser.parse_args()
    if not os.path.isfile(args.dict):
        raise RuntimeError('**Error! Cant open dict %s' % args.dict)
    if not os.path.isdir(args.model_dir):
        raise RuntimeError('**Error! Cant open model dir %s' % args.model_dir)
    return args


def main():

    args = parse_args()
    model_type, g2i, p2i = read_meta('%s/meta' % args.model_dir)
    G2PModel, hparams = import_model_type(model_type)
    with open('%s/hparams' % args.model_dir, 'r') as infp:
        loaded = json.load(infp)
        hparams.parse_json(loaded)
    d = read_dict(args.dict)
    d = encode_dict(d, g2i, p2i)
    i2p = {v: k for k, v in p2i.items()}
    model = G2PModel(hparams, is_training=False, with_target=True, reuse=False)
    print('**Info: model created')

    sess = tf.Session()
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(args.model_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    model_path = os.path.join(args.model_dir, 'g2p-%d' % args.restore)
    saver.restore(sess, model_path)

    d_batched = group_batch_pad(d, g2i, p2i, hparams.group_size,
                                hparams.batch_size, model_type)
    print('**Info: data grouped and batched')

    wer, stressless_wer, eval_took = compute_wer(sess, model, d_batched, i2p,
                                                 model_type)
    print(' Eval: wer %f; stressless wer %f; eval took %f' %
          (wer, stressless_wer, eval_took))


if __name__ == '__main__':
    main()
