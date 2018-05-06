
TFMAKEFILE_DIR=./tensorflow/tensorflow/contrib/makefile

all:
	gcc -O3 --std=c++11 g2p_app.cc g2p.cc \
		./phonetisaurus/local/lib/python2.7/site-packages/Phonetisaurus.a \
		-I. -I./tensorflow/bazel-tensorflow/ -I./tensorflow/bazel-genfiles/ \
		-I$(TFMAKEFILE_DIR)/downloads/eigen \
		-I$(TFMAKEFILE_DIR)/gen/protobuf-host/include \
		-I./phonetisaurus/local/include/ \
		-I./phonetisaurus/src/3rdparty/utfcpp \
		-I./phonetisaurus/src/ \
		-L./tensorflow/bazel-bin/tensorflow \
		-L./phonetisaurus/local/lib \
		-Wl,--allow-multiple-definition -Wl,--whole-archive \
		-ltensorflow_all -ltensorflow_framework -lfst \
		-lstdc++ -ldl -lpthread -lm -lz -o g2p_app
