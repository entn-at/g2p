
import re
import sys
import string

def read_dict(path):
    d = []
    with open(path, 'r') as infp:
        for line in infp:
            line = line.strip()
            # drop comments
            if line.startswith(';;;'):
                continue
            # drop inline comments (as in amepd)
            if '#' in line:
                line = line.split('#')[0].rstrip()
            parts = line.split()
            word = parts[0]
            pron = parts[1:]
            d.append((word, pron))
    return d


def write_dict(d, path):
    with open(path, 'w') as outfp:
        for word, pron in d:
            outfp.write('%s\t%s\n' % (word, ' '.join(pron)))




def filter_long_pron(d, diff=3):
    d_filt = []
    for w, p in d:
        if len(p) - len(w) < diff:
            d_filt.append((w, p))
    return d_filt


def filter_short_words(d, word_len=1):
    d_filt = []
    for w, p in d:
        if len(w) > word_len:
            d_filt.append((w, p))
    return d_filt


def filter_rare_graphemes(d, threshold=100):
    freq = {}
    for w, _ in d:
        for c in w:
            if c in freq:
                freq[c] += 1
            else:
                freq[c] = 1
    to_drop = []
    for c, v in freq.items():
        if v < threshold:
            to_drop.append(c)
    to_drop = set(to_drop)

    d_filt = []
    for w, p in d:
        to_add = True
        for c in w:
            if c in to_drop:
                to_add = False
                break
        if to_add:
            d_filt[w] = p

    return d_filt


def filter_word_idx(d):
    d_filt = []
    for w, p in d:
        if '(' in w:
            w = w.split('(')[0]
        d_filt[w] = p
    return d_filt


def fix_case(d):
    d_fixed = []
    for w, p in d:
        d_fixed.append((w.lower(), [x.upper() for x in p]))
    return d_fixed


def split_train_dev_test(path, train_path, dev_path, test_path, dev_every_nth=20,
                         test_every_nth=10):
    d = read_dict(path)
    d = filter_word_idx(d)
    d = filter_long_pron(d)
    d = filter_short_words(d)
    d = fix_case(d)
    d = filter_rare_graphemes(d)

    # split the dict into train/dev/test
    traind = []
    devd = []
    testd = []
    for i, pair in enumerate(d):
        if (i + 1) % dev_every_nth == 0:
            devd.append(pair)
        elif (i + 2) % test_every_nth == 0:
            testd.append(pair)
        else:
            traind.append(pair)

    write_dict(traind, train_path)
    write_dict(devd, dev_path)
    write_dict(testd, test_path)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise RuntimeError('Usage: <indict> <train-dict> <dev-dict> <test-dict>')

    split_train_dev_test(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
