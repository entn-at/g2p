
import os
import time
import json
import argparse

import tensorflow as tf

from read_utils import *
from nn.encode_utils import *
from nn.utils import *


def parse_args():
    arg_parser = argparse.ArgumentParser(description='Trains G2P model')
    arg_parser.add_argument('--train', required=True,
                            help='Dictionary to train on')
    arg_parser.add_argument('--dev', help='Dictionary to eval on')
    arg_parser.add_argument('--model-dir', default='g2p_model',
                            help='Directory to put model to')
    arg_parser.add_argument('--hparams', default='', help='Overwrites hparams')
    arg_parser.add_argument('--restore', type=int,
                            help='Step to restore if any')
    arg_parser.add_argument('--model-type',
                            choices=('ctc', 'attention', 'transformer', 'transformer_ctc'),
                            default='ctc', help='What kind of model to train')

    args = arg_parser.parse_args()
    if not os.path.isfile(args.train):
        raise RuntimeError('**Error! Cant open dict %s' % args.train)
    if args.dev and not os.path.isfile(args.dev):
        raise RuntimeError('**Error! Cant open dict %s' % args.dev)
    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)
    return args


class AverageWindow:
    def __init__(self, capacity=100):
        self.loss = []
        self.speed = []
        self.capacity = capacity

    def add(self, loss, speed):
        self.loss.append(loss)
        self.speed.append(speed)
        if len(self.loss) > self.capacity:
            self.loss = self.loss[1:]
            self.speed = self.speed[1:]

    def get(self):
        loss = sum(self.loss) / float(len(self.loss))
        speed = sum(self.speed) / float(len(self.speed))
        return loss, speed


def main():
    args = parse_args()

    G2PModel, hparams = import_model_type(args.model_type)
    hparams.parse(args.hparams)
    with open('%s/hparams' % args.model_dir, 'w') as outfp:
        json.dump(hparams.to_json(), outfp)

    d = read_cmudict(args.train)
    g2i = get_graphemes_map(d)
    p2i = get_phonemes_map(d)
    write_meta(args.model_type, g2i, p2i, '%s/meta' % args.model_dir)

    traind = encode_dict(d, g2i, p2i)
    if args.dev:
        d = read_cmudict(args.dev)
        devd = encode_dict(d, g2i, p2i)

    print('**Info: training inputs read. There are %d graphemes and %d phonemes' %
          (len(g2i), len(p2i)))
    i2p = {v: k for k, v in p2i.items()}

    model = G2PModel(hparams, is_training=True, with_target=True,
                     reuse=False)
    model.add_loss()
    model.add_train_and_stats()
    if args.dev:
        dev_model = G2PModel(hparams, is_training=False, with_target=True,
                             reuse=True)
    print('**Info: model created')

    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=0)
    summary_writer = tf.summary.FileWriter(args.model_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    model_prefix = os.path.join(args.model_dir, 'g2p')
    if args.restore:
        model_path ='%s-%d' % (model_prefix, args.restore)
        saver.restore(sess, model_path)

    d_batched = group_batch_pad(traind, g2i, p2i, hparams.group_size,
                                hparams.batch_size, args.model_type)
    if args.dev:
        ddev_batched = group_batch_pad(devd, g2i, p2i, hparams.group_size,
                                       hparams.batch_size, args.model_type)

    print('**Info: data grouped and batched')

    step = 0
    epoch = 0
    accum = AverageWindow()
    while step < hparams.max_steps:
        for batch in d_batched:
            start_time = time.time()
            step, loss, _, summary = sess.run(
                    [model.global_step, model.loss, model.train_op, model.stats_op],
                    feed_dict=model.create_feed_dict(batch))
            summary_writer.add_summary(summary, step)
            step_time = time.time() - start_time
            accum.add(loss, step_time)
            loss, step_time = accum.get()
            sys.stdout.write('\rTrain: step %d; loss %f; step time %f' %
                             (step, loss, step_time))

            if step % hparams.save_every_nth == 0:
                saver.save(sess, model_prefix, global_step=step)

            if step % hparams.eval_every_nth == 0 and args.dev:
                wer, stressless_wer, eval_took = compute_wer(
                        sess, dev_model, ddev_batched, i2p, args.model_type)
                print(' Eval: step %d; wer %f; stressless wer %f; eval took %f' %
                      (step, wer, stressless_wer, eval_took))

                summary_writer.add_summary(
                    tf.Summary(value=[tf.Summary.Value(tag='wer', simple_value=wer)]),
                    global_step=step)
                summary_writer.add_summary(
                    tf.Summary(value=[tf.Summary.Value(tag='eval_speed', simple_value=eval_took)]),
                    global_step=step)
                summary_writer.add_summary(
                    tf.Summary(value=[tf.Summary.Value(tag='stressless_wer', simple_value=stressless_wer)]),
                    global_step=step)

        epoch += 1
        if epoch % hparams.reshuffle_every_nth == 0:
            d_batched = group_batch_pad(traind, g2i, p2i, hparams.group_size,
                                        hparams.batch_size, args.model_type)


if __name__ == '__main__':
    main()
