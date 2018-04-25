import re
from random import shuffle
import numpy as np
import collections


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


def pad_arr_to(arr, stop_symbol, expected_len):
    padding = expected_len - len(arr)
    assert(padding >= 0)
    if padding == 0:
        return arr
    return np.pad(arr, (0, padding), 'constant',
                  constant_values=stop_symbol)


def encode_dict(d, g2i, p2i):
    newd = []
    for word, pron in d:
        word = encode(word, g2i)
        pron = encode(pron, p2i)
        newd.append((word, pron))
    return newd


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


def _preprocess_ctc_batch(batch, grapheme_pad):
    words = [x[0] for x in batch]
    prons = [x[1] for x in batch]
    # to ensure that grapheme sequence is longer than phoneme seq
    words = [np.pad(words, (0, 3), 'constant', constant_values=grapheme_pad) for w in words]
    for w, p in zip(words, prons):
        assert len(w) >= len(p)
    seq_lens = [len(x) for x in words]
    max_len = max(seq_lens)
    words = np.stack([pad_arr_to(x, grapheme_pad, max_len) for x in words])
    # as for pronunciations - that's a bit more complicated, those should be prepared
    # as sparse tensors
    prons = _create_sparse_tuple(prons)
    return words, seq_lens, prons


def _preprocess_attention_batch(batch, grapheme_pad, phoneme_start, phoneme_pad):
    words = [x[0] for x in batch]
    prons = [x[1] for x in batch]
    #words = [np.pad(w, (0, 1), 'constant', constant_values=grapheme_pad) for w in words]
    prons = [np.pad(p, (0, 1), 'constant', constant_values=phoneme_pad) for p in prons]

    words_len = [len(x) for x in words]
    prons_len = [len(x) for x in prons]

    prons = [np.pad(p, (1, 0), 'constant', constant_values=phoneme_start) for p in prons]

    max_word_len = max(words_len)
    max_pron_len = max(prons_len) + 1

    words = np.stack([pad_arr_to(x, grapheme_pad, max_word_len) for x in words])
    prons = np.stack([pad_arr_to(x, phoneme_pad, max_pron_len) for x in prons])

    return words, words_len, prons, prons_len


def group_batch(d, group_size, batch_size):
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
    return d


def group_batch_pad_ctc(d, grapheme_pad, group_size, batch_size):
    d = group_batch(d, group_size, batch_size)
    # pad batches so the length is the same within the batch
    d = [_preprocess_ctc_batch(b, grapheme_pad) for b in d]
    return d


def group_batch_pad_attention(d, grapheme_pad, phoneme_start, phoneme_pad,
                              group_size, batch_size):
    d = group_batch(d, group_size, batch_size)
    d = [_preprocess_attention_batch(b, grapheme_pad, phoneme_start, phoneme_pad) for b in d]
    return d


def group_batch_pad(d, g2i, p2i, group_size, batch_size, pad_type):
    if pad_type == 'ctc':
        return group_batch_pad_ctc(d, len(g2i), group_size, batch_size)
    elif pad_type == 'attention' or pad_type == 'transformer':
        return group_batch_pad_attention(d, len(g2i), len(p2i), len(p2i) + 1,
                                         group_size, batch_size)


def decode_pron(i2p, pron, is_sparse=True):
    if is_sparse:
        indices, values, shape = pron
        arr = np.ones(shape) * len(i2p)
        arr[indices[:, 0], indices[:, 1]] = values
        pron = arr
    decoded = []
    for encoded in pron:
        decoded.append([i2p[x] for x in encoded if x < len(i2p)])
    return decoded


