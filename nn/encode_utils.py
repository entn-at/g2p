import re
from random import shuffle
import numpy as np
import collections


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
            d.append((parts[0], parts[1:]))
    return d


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


def filter_suffix(d):
    return [(drop_word_idx(w), p) for w, p in d]


def _set_to_map(s):
    l = sorted(list(s))
    m = collections.OrderedDict()
    for i, val in enumerate(l):
        m[val] = i
    return m


def get_graphemes_map(d):
    gset = set()
    for w, _ in d:
        gset.update(drop_word_idx(w))
    return _set_to_map(gset)


def get_phonemes_map(d):
    pset = set()
    for _, p in d:
        pset.update(p)
    return _set_to_map(pset)


def encode(l, m):
    return np.array([m[element] for element in l])


def pad_arr(arr, stop_symbol, padding=3):
    return np.pad(arr, (0, padding), 'constant',
                  constant_values=stop_symbol)


def pad_arr_to(arr, stop_symbol, expected_len):
    padding = expected_len - len(arr)
    assert(padding >= 0)
    if padding == 0:
        return arr
    return pad_arr(arr, stop_symbol, padding=padding)


def encode_dict(d, g2i, p2i):
    newd = []
    for word, pron in d:
        word = encode(word, g2i)
        pron = encode(pron, p2i)
        word = pad_arr(word, len(g2i), padding=3)
        diff = len(word) - len(pron)
        assert(diff >= 0)
        newd.append((word, pron))
    return newd


def get_training_inputs(path, dev_every_n, mapping=None):
    d = read_cmudict(path)
    d = filter_long_pron(d)
    d = filter_short_words(d)
    d = filter_noneng(d)
    d = filter_suffix(d)

    # get grapheme2index and phoneme2index mappings
    if mapping:
        g2i, p2i = mapping
    else:
        g2i = get_graphemes_map(d)
        p2i = get_phonemes_map(d)

    # split the dict into train/dev
    traind = []
    devd = []
    for i, pair in enumerate(d):
        if (i + 1) % dev_every_n == 0:
            devd.append(pair)
        else:
            traind.append(pair)

    traind = encode_dict(traind, g2i, p2i)
    devd = encode_dict(devd, g2i, p2i)

    return traind, devd, g2i, p2i


def _create_sparse_tuple(prons):
    indices = []
    values = []
    for n, pron in enumerate(prons):
        indices.extend(zip([n]*len(pron), range(len(pron))))
        values.extend(pron)
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.int32)
    shape = np.asarray([len(prons), np.asarray(indices).max(0)[1]+1], dtype=np.int64)
    return indices, values, shape


def _preprocess_batch(batch, grapheme_pad_symbol):
    words = [x[0] for x in batch]
    prons = [x[1] for x in batch]
    seq_lens = [len(x) for x in words]
    max_len = max(seq_lens)
    words = np.stack([pad_arr_to(x, grapheme_pad_symbol, max_len) for x in words])
    # as for pronunciations - that's a bit more complicated, those should be prepared
    # as sparse tensors
    prons = _create_sparse_tuple(prons)
    return words, seq_lens, prons


def group_batch_pad(d, grapheme_pad, group_size=32, batch_size=32):
    total_group_size = group_size * batch_size
    # shuffle pairs (word - pronunciation)
    shuffle(d)
    # split into groups
    d = [d[i:i+total_group_size] for i in range(0, len(d), total_group_size)]
    # sort within the groups
    d = [sorted(g, key=lambda x: len(x[0])) for g in d]
    # split the groups into batches
    d = [g[i:i+batch_size] for g in d for i in range(0, len(g), batch_size)]
    # shuffle the batches
    shuffle(d)
    # pad batches so the length is the same within the batch
    d = [_preprocess_batch(b, grapheme_pad) for b in d]
    return d


def decode_pron(i2p, pron):
    indices, values, shape = pron
    arr = np.ones(shape) * len(i2p)
    arr[indices[:, 0], indices[:, 1]] = values
    decoded = []
    for encoded in arr:
        decoded.append([i2p[x] for x in encoded if x < len(i2p)])
    return decoded


