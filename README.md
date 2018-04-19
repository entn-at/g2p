# G2P (Grapheme-to-phoneme)

Repo with code for english pronunciation generation

Performance on amepd (using split by read_utils) in WER / stressless WER:

* phonetisaurus 8-gram: 0.3591 / 0.2906
* phonetisaurus 5-gram: 0.3692 / 0.2937
* cmu seq2seq g2p: 0.3194 / 0.27877

At this point - struggling to achieve cmu seq2seq level. Seems to be problems with regularization.
