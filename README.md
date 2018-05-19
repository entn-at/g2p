# G2P (Grapheme-to-phoneme)

Repo with code for pronunciation generation. Several approaches are implemented and evaluated on different open-source dictionaries. To my knowledge this is the most accurate g2p on the web! You can try it out here: [__demo__](http://159.69.1.31/)

## Why?

There is a nice [neural g2p from cmu](https://github.com/cmusphinx/g2p-seq2seq), but the dependency on tensor2tensor was really disturbing for me. It is also lacking intersection with FST output as in [paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43264.pdf). As a bonus - comparison of transformer architecture to regular encoder-decoder with attention and CTC. From engineering point of view, important aspects are shown: tensorflow inference in C++, integration of g2p to C++/python applications, python g2p web-service.

## What?

Following techniques are implemented:

* **FST**: strong baseline is [phonetisaurus](https://github.com/AdolfVonKleist/Phonetisaurus). It can also improve neural g2p.
* **CTC**: connectionist temporal classification as in original paper.
* **encoder-decoder with attention**: classic architecture for seq2seq modelling. All the inspiration was taken from this [tutorial](https://github.com/tensorflow/nmt)
* **transformer**: state-of-art architecture for machine translation, described in this [paper](https://arxiv.org/pdf/1706.03762.pdf). Implementation was borrowed from this [repo](https://github.com/Kyubyong/transformer), inference was improved with a while loop.
* **transformer + FST**: outputs of neural and joint n-gram models could be combined by performing "intersection" operation from FST theory. 

## Performance
Performance was evaluated for several dictionaries: 

* American pronunciation dictionary [link](https://github.com/rhdunn/amepd)
	
	approach                 | WER, %       | stressless WER, %
	-------------------------|--------------|-------------------
	fst 5-gram               | 0.3692       | 0.2937
	fst 8-gram               | 0.3589       | 0.2899
	cmu seq2seq g2p          | 0.3087       | 0.2708
	ctc                      | 0.3293       | 0.2863
	attention                | 0.3302       | 0.2892
	transformer              | 0.3027       | 0.2685
	transformer + fst 8-gram | __0.2987__   | __0.2575__
	
* CMU dict [link](https://github.com/cmusphinx/cmudict)

	approach                 | WER, %       | stressless WER, %
	-------------------------|--------------|-------------------
	fst 8-gram               | 0.4053       | 0.3330
	transformer              | 0.3581       | 0.3137
	transformer + fst 8-gram | 0.3520       | 0.3026

* Beep dict [link](http://svr-www.eng.cam.ac.uk/comp.speech/Section1/Lexical/beep.html)

	approach                 | stressless WER, %
	-------------------------|-------------------
	fst 8-gram               | 0.2114
	transformer              | __0.1898__
	transformer + fst 8-gram | 0.1944

* Russian dict by nshm [link](https://sourceforge.net/projects/cmusphinx/files/Acoustic%20and%20Language%20Models/Russian/)

	approach                 | WER, %       | stressless WER, %
	-------------------------|--------------|-------------------
	fst 8-gram               | 0.1803       | 0.0079
	transformer              | __0.1256__   | __0.0073__
	transformer + fst 8-gram | 0.1304       | 0.0073

Be aware! Not all of them are suitable for commerical usage.

## Constrained g2p
In real-life applications, combining FST and nural g2p to gain couple of percents might seem an overkill, but for if there is a memory constrain, neural g2p still has something to offer. Some results for amepd:

approach    | model size |  WER, %       | stressless WER, %
------------|------------|---------------|-------------------
fst 3-gram  | 1.9mB      | 0.4829        | 0.3812
small ctc   | 1.1mB      | 0.3608        | 0.3108

## TODO
Possible improvements:

* Improve transformer implementation with caching during the inference
* More experimentation with model hyper-params, same results should be obtained with smaller and faster NN model.
* Optimize the trained networks: [documentation](https://www.tensorflow.org/mobile/optimizing)
* In case of fst+nn, the most time is taken by intersection operation. During lattice composition from NN output, it can be sufficiently pruned, speeding up subsequent intersection.
* More languages!

## Help
* You can __get help__ e-mailing to <bicuser470@gmail.com> or opening an issue
* You can __help__ pointing to the pronunciation dictionaries, sharing more powerful host for demo, donating:

[![paypal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=MAJ6WMVD2PRXS)

