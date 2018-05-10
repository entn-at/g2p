
import os
import sys
import argparse

from libg2p import PyG2P

def parse_args():
    arg_parser = argparse.ArgumentParser(
            description='Produces pronunciation for piped words')
    arg_parser.add_argument('--nn', default='', help='Path to nn model')
    arg_parser.add_argument('--nn-meta', default='', help='Path to nn meta')
    arg_parser.add_argument('--fst', default='', help='Path to fst')
    args = arg_parser.parse_args()
    if not args.nn and not args.fst:
        raise RuntimeError('**Error! Either nn or fst or both should be provided')
    if args.nn and not args.nn_meta:
        raise RuntimeError('**Error! You should provide meta to use nn')
    if args.nn and (not os.path.isfile(args.nn) or not os.path.isfile(args.nn_meta)):
        raise RuntimeError('**Error! Cant open nn or meta')
    if args.fst and not os.path.isfile(args.fst):
        raise RuntimeError('**Error! Cant open fst')
    return args


def main():
    args = parse_args()
    g2p = PyG2P(args.nn, args.nn_meta, args.fst)
    while True:
        try:
            word = sys.stdin.readline().strip()
        except KeyboardInterrupt:
            break

        if not word:
            break

        print(g2p.Phonetisize(word))


if __name__ == '__main__':
    main()
