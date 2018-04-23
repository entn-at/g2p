#!/usr/bin/env python2.7

import os
import time
import argparse

import phonetisaurus

from read_utils import *

def parse_args():
    arg_parser = argparse.ArgumentParser(description='Evaluates fst given dict')
    arg_parser.add_argument('--model', required=True,
                            help='Path to model for eval')
    arg_parser.add_argument('--dict', required=True,
                            help='Path to dict to run eval for')
    args = arg_parser.parse_args()
    if not os.path.isfile(args.model):
        raise RuntimeError('**Error! failed to open model %s' % args.model)
    if not os.path.isfile(args.dict):
        raise RuntimeError('**Error! failed to open dict %s' % args.dict)
    return args

def main():
    args = parse_args()
    eval_start = time.time()
    model = phonetisaurus.Phonetisaurus(args.model)
    d = read_cmudict(args.dict)
    wer = 0.0
    stressless_wer = 0.0
    for word, pron in d:
        # token, beam, threshold, write decoded fsts to disc,
        # accum probs across unique pronunciations, target prob mass
        results = model.Phoneticize(word, 1, 500, 10., False, False, 0.0)
        result = next(iter(results))
        pred_pron = [model.FindOsym(u) for u in result.Uniques]
        oo = ' '.join(pron)
        pp = ' '.join(pred_pron)
        if oo != pp:
            wer += 1
        oo = ' '.join(drop_stress(pron))
        pp = ' '.join(drop_stress(pred_pron))
        if oo != pp:
            stressless_wer += 1

    wer /= float(len(d))
    stressless_wer /= float(len(d))
    eval_took = time.time() - eval_start

    print(' Eval: wer %f; stressless wer %f; eval took %f' %
          (wer, stressless_wer, eval_took))


if __name__ == '__main__':
    main()
