#!/usr/bin/bash

if [ "$#" -ne 2 ]; then
	echo "usage: <lexicon> <out-file>"
	exit 1
fi

./phonetisaurus/local/bin/phonetisaurus-train --lexicon $1 --seq2_del --seq1_del --ngram_order 8 --dir_prefix . --model_prefix $2
