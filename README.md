# G2P (Grapheme-to-phoneme)

Repo with code for english pronunciation generation

Performance on amepd (using split by read_utils) in WER / stressless WER:

* phonetisaurus 8-gram: 0.3591 / 0.2906
* phonetisaurus 5-gram: 0.3692 / 0.2937
* cmu seq2seq g2p: 0.3194 / 0.2788
* ctc:  0.3298 / 0.2872
* attention: 0.3302 / 0.2892
