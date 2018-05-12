# G2P (Grapheme-to-phoneme)

Repo with code for english pronunciation generation

Performance on amepd (using split by read_utils) in WER / stressless WER:

* phonetisaurus 5-gram: 0.3692 / 0.2937
* phonetisaurus 8-gram: 0.3589 / 0.2899
* cmu seq2seq g2p: 0.3087 / 0.2708
* ctc:  0.3293 / 0.2863
* attention: 0.3302 / 0.2892
* transformer: 0.3029 / 0.2682
* transformer + 8-gram fst: 0.2993 / 0.2574

Performance on cmudict in WER / stressless WER:

* phonetisaurus 8-gram: 0.405269 / 0.332987

Performance on beep (net-talk) in stressless WER:

* phonetisaurus 8-gram: 0.211417

Performance on runshm in WER / stressless WER:

* phonetisaurus 8-gram: 0.180319 / 0.007887
