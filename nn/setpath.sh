
if [[ $_ == $0 ]]; then
	echo "**Error! You should 'source' this script"
	exit 1
fi

export LD_LIBRARY_PATH=$PWD/tensorflow/bazel-bin/tensorflow:$LD_LIBRARY_PATH
