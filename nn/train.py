
import os
import sys
import time
import argparse

import tensorflow as tf

from read_utils import *
from nn.hparams import hparams
from nn.model import G2PModel, create_placeholders
from nn.encode_utils import *


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
    hparams.parse(args.hparams)

    d = read_cmudict(args.train)
    g2i = get_graphemes_map(d)
    p2i = get_phonemes_map(d)
    traind = encode_dict(d, g2i, p2i)
    if args.dev:
        d = read_cmudict(args.dev)
        devd = encode_dict(d, g2i, p2i)
    assert(len(g2i) + 1 == hparams.graphemes_num)
    assert(len(p2i) + 1 == hparams.phonemes_num)
    print('**Info: training inputs read. There are %d graphemes and %d phonemes' %
          (len(g2i), len(p2i)))
    i2p = {v: k for k, v in p2i.items()}

    inputs, input_lengths, targets = create_placeholders(with_target=True)
    model = G2PModel(inputs, input_lengths, hparams, is_training=True,
                     reuse=False)
    model.add_decoder()
    model.add_loss(targets)
    model.add_train_and_stats()

    if args.dev:
        dev_model = G2PModel(inputs, input_lengths, hparams, is_training=False,
                             reuse=True)
        dev_model.add_decoder()
        dev_model.add_loss(targets)
    print('**Info: model created')

    sess = tf.Session()
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(args.model_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    model_prefix = os.path.join(args.model_dir, 'g2p')
    if args.restore:
        model_path ='%s-%d' % (model_prefix, args.restore)
        saver.restore(sess, model_path)

    d_batched = group_batch_pad(traind, len(g2i),
                                group_size=hparams.group_size,
                                batch_size=hparams.batch_size)
    print(len(d_batched))
    if args.dev:
        ddev_batched = group_batch_pad(devd, len(g2i),
                                       group_size=hparams.group_size,
                                       batch_size=hparams.batch_size)
    print('**Info: data grouped and batched')

    step = 0
    epoch = 0
    accum = AverageWindow()
    while step < hparams.max_steps:
        for words, input_len, pron in d_batched:
            start_time = time.time()
            step, loss, _, summary = sess.run(
                    [model.global_step, model.loss, model.train_op, model.stats_op],
                    feed_dict={inputs: words, input_lengths: input_len,
                               targets: pron})
            summary_writer.add_summary(summary, step)
            step_time = time.time() - start_time
            accum.add(loss, step_time)
            loss, step_time = accum.get()
            sys.stdout.write('\rTrain: step %d; loss %f; step time %f' %
                             (step, loss, step_time))

            if step % hparams.save_every_nth == 0:
                saver.save(sess, model_prefix, global_step=step)

            if step % hparams.eval_every_nth == 0 and args.dev:
                average_loss = 0.0
                wer = 0.0
                stressless_wer = 0.0
                words_num = 0
                eval_start = time.time()
                for words, input_len, pron in ddev_batched:
                    loss, output = sess.run([dev_model.loss, dev_model.decoded[0]],
                                            feed_dict={inputs: words,
                                                       input_lengths: input_len,
                                                       targets: pron})
                    average_loss += loss

                    orig = decode_pron(i2p, pron)
                    predicted = decode_pron(i2p, output)

                    words_num += len(orig)
                    for o, p in zip(orig, predicted):
                        oo = ' '.join(o)
                        pp = ' '.join(p)
                        if oo != pp:
                            wer += 1
                        oo = ' '.join(drop_stress(o))
                        pp = ' '.join(drop_stress(p))
                        if oo != pp:
                            stressless_wer += 1

                average_loss /= len(ddev_batched)
                wer /= float(words_num)
                stressless_wer /= float(words_num)
                eval_took = time.time() - eval_start
                print(' Eval: step %d; loss %f; wer %f; stressless wer %f; eval took %f' %
                      (step, average_loss, wer, stressless_wer, eval_took))

        epoch += 1
        if epoch % hparams.reshuffle_every_nth == 0:
            d_batched = group_batch_pad(traind, len(g2i),
                                        group_size=hparams.group_size,
                                        batch_size=hparams.batch_size)


if __name__ == '__main__':
    main()
