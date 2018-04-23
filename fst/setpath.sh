
if [[ $_ == $0 ]]; then
	echo "**Error! You should 'source' this script"
	exit 1
fi

export LD_LIBRARY_PATH=$PWD/phonetisaurus/local/lib:$LD_LIBRARY_PATH
export PATH=$PWD/phonetisaurus/local/bin:$PATH
export PYTHONPATH=$PWD/phonetisaurus/local/lib/python2.7/site-packages:$PWD/phonetisaurus/python/:$PYTHONPATH
