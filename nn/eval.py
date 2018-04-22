
import os
import sys
import time
import json
import string
import argparse

import tensorflow as tf

from nn.encode_utils import *
from read_utils import *


def parse_args():
    arg_parser = argparse.ArgumentParser(description='Trains G2P model')
    arg_parser.add_argument('--dict', required=True,
                            help='Dictionary to train on')
    arg_parser.add_argument('--model-dir', default='g2p_model', required=True,
                            help='Directory to put model to')
    arg_parser.add_argument('--hparams', default='', help='Overwrites hparams')
    arg_parser.add_argument('--restore', type=int, required=True,
                            help='Step to restore from')
    arg_parser.add_argument('--model-type', choices=('ctc', 'attention'),
                            help='Type of model')

    args = arg_parser.parse_args()
    if not os.path.isfile(args.dict):
        raise RuntimeError('**Error! Cant open dict %s' % args.dict)
    if not os.path.isdir(args.model_dir):
        raise RuntimeError('**Error! Cant open model dir %s' % args.model_dir)
    return args


def main():
    args = parse_args()

    if args.model_type == 'ctc':
        from nn.hparams_ctc import hparams
        from nn.model_ctc import G2PModel
    elif args.model_type == 'attention':
        from nn.hparams_attention import hparams
        from nn.model_attention import G2PModel

    hparams.parse(args.hparams)
    with open('%s/g2i.json' % args.model_dir, 'r') as infp:
        g2i = json.load(infp)
    with open('%s/p2i.json' % args.model_dir, 'r') as infp:
        p2i = json.load(infp)
    d = read_cmudict(args.dict)
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
                                hparams.batch_size, args.model_type)
    print('**Info: data grouped and batched')

    wer = 0.0
    stressless_wer = 0.0
    words_num = 0
    eval_start = time.time()
    for batch in d_batched:
        output = sess.run(model.decoded_best,
                          feed_dict=model.create_feed_dict(batch))

        orig = decode_pron(i2p, batch[2], is_sparse=args.model_type == 'ctc')
        predicted = decode_pron(i2p, output, is_sparse=args.model_type == 'ctc')

        words_num += len(orig)
        for o, p in zip(orig, predicted):
            oo = ' '.join(o)
            pp = ' '.join(p)
            if oo != pp:
                wer += 1
            oo = ' '.join([x.rstrip(string.digits) for x in o])
            pp = ' '.join([x.rstrip(string.digits) for x in p])
            if oo != pp:
                stressless_wer += 1

    wer /= float(words_num)
    stressless_wer /= float(words_num)
    eval_took = time.time() - eval_start
    print(' Eval: wer %f; stressless wer %f; eval took %f' %
          (wer, stressless_wer, eval_took))


if __name__ == '__main__':
    main()
