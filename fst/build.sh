#!/usr/bin/env bash

pushd phonetisaurus
wget http://www.openslr.org/resources/2/openfst-1.6.2.tar.gz
tar -zxvf openfst-1.6.2.tar.gz
pushd openfst-1.6.2
mkdir ../local
./configure --enable-static --enable-shared --enable-far --enable-ngram-fsts \
	--prefix=$PWD/../local
make -j 8
make install
popd
rm -f openfst-1.6.2.tar.gz

# you may need to run:
# sudo apt-get install autoconf-archive
git clone https://github.com/mitlm/mitlm.git
pushd mitlm
./autogen.sh --prefix=$PWD/../local
make -j 8
make install
popd

export LD_LIBRARY_PATH=$PWD/local/lib:$LD_LIBRARY_PATH
patch -p1 < ../fst/phonetisaurus_patch.patch
./configure --with-openfst-libs=$PWD/local/lib \
	--with-openfst-includes=$PWD/local/include \
	--enable-python --prefix=$PWD/local/
make -j 8 all
make install
popd
