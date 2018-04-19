
import re
import sys
import string

def read_cmudict(path):
    d = []
    with open(path, 'r') as infp:
        for line in infp:
            line = line.strip()
            if line.startswith(';;;'):
                continue
            if '#' in line:
                line = line.split('#')[0].rstrip()
            parts = line.split()
            word = parts[0]
            pron = [x for x in parts[1:] if x != '-']
            d.append((word, pron))
    return d


def write_cmudict(d, path):
    with open(path, 'w') as outfp:
        for word, pron in d:
            outfp.write('%s\t%s\n' % (word, ' '.join(pron)))


def drop_word_idx(w):
    if '(' in w:
        return w.split('(')[0]
    else:
        return w


def filter_long_pron(d, diff=3):
    d_filt = []
    for w, p in d:
        if len(p) - len(drop_word_idx(w)) < diff:
            d_filt.append((w, p))
    return d_filt


def filter_short_words(d, word_len=1):
    d_filt = []
    for w, p in d:
        if len(drop_word_idx(w)) > word_len:
            d_filt.append((w, p))
    return d_filt


def filter_noneng(d):
    d_filt = []
    eng_graphemes_re = re.compile('^[A-Z\']+$')
    for w, p in d:
        w_noidx = drop_word_idx(w)
        if eng_graphemes_re.match(w_noidx):
            d_filt.append((w, p))
    return d_filt


def drop_stress(phonemes):
    return [x.rstrip(string.digits) for x in phonemes]


def filter_stress(d):
    d_filt = []
    for w, p in d:
        p = drop_stress(p)
        d_filt.append((w, p))
    return d_filt


def filter_suffix(d):
    return [(drop_word_idx(w), p) for w, p in d]


def split_train_dev_test(path, train_path, dev_path, test_path, dev_every_nth=20,
                         test_every_nth=10):
    d = read_cmudict(path)
    d = filter_long_pron(d)
    d = filter_short_words(d)
    d = filter_noneng(d)
    d = filter_suffix(d)

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

    write_cmudict(traind, train_path)
    write_cmudict(devd, dev_path)
    write_cmudict(testd, test_path)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise RuntimeError('Usage: <indict> <train-dict> <dev-dict> <test-dict>')

    split_train_dev_test(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
