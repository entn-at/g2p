
TFMAKEFILE_DIR=./tensorflow/tensorflow/contrib/makefile

all: pyx lib app

pyx:
	cython -3 --cplus libg2p.pyx

lib:
	gcc -O3 -fPIC -std=c++11 \
		g2p.cc phonetisaurus/src/lib/util.cc libg2p.cpp \
		-I. -I./tensorflow/bazel-tensorflow/ -I./tensorflow/bazel-genfiles/ \
		-I$(TFMAKEFILE_DIR)/downloads/eigen \
		-I$(TFMAKEFILE_DIR)/gen/protobuf-host/include \
		-I./phonetisaurus/local/include/ \
		-I./phonetisaurus/src/3rdparty/utfcpp \
		-I./phonetisaurus/src/ \
		`python3-config --includes` \
		-L./tensorflow/bazel-bin/tensorflow \
		-L./phonetisaurus/local/lib \
		-L/usr/lib/python3.5/config-3.5m-x86_64-linux-gnu \
		-Wl,--allow-multiple-definition -Wl,--whole-archive -ltensorflow_all \
		-ltensorflow_framework -Wl,--no-whole-archive \
		-lfst -lstdc++ -ldl -lpthread -lm -lz -lpython3.5 \
		-shared -o libg2p.so

app:
	gcc -O3 -std=c++11 g2p_app.cc -L. -lg2p -lstdc++ -o g2p_app
