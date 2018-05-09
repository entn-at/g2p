# G2P (Grapheme-to-phoneme)

Repo with code for english pronunciation generation

Performance on amepd (using split by read_utils) in WER / stressless WER:

* phonetisaurus 5-gram: 0.3692 / 0.2937
* phonetisaurus 8-gram: 0.3591 / 0.2906
* cmu seq2seq g2p: 0.3087 / 0.2708
* ctc:  0.3293 / 0.2863
* attention: 0.3302 / 0.2892
* transformer: 0.3093 / 0.2715
* transformer + 8-gram fst: 0.3025 / 0.2588
