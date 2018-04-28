#!/usr/bin/env python

import os
import sys
import re
import json
import argparse
from shutil import copyfile

import tensorflow as tf
from tensorflow.core.example import example_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import freeze_graph
from tensorflow.python.training import saver as saver_lib

from nn.utils import *


def freeze_model(model_path, freeze_dir, output_graph_name):

    name = re.sub(r'-\d+$', '', os.path.basename(model_path))
    model_dir = os.path.dirname(model_path)
    model_type, _, _ = read_meta('%s/meta' % model_dir)

    if not os.path.isdir(freeze_dir):
        os.makedirs(freeze_dir)
    copyfile('%s/meta' % model_dir, '%s/meta' % freeze_dir)
    checkpoint_prefix = os.path.join(freeze_dir, "saved_checkpoint")
    checkpoint_state_name = "checkpoint_state"
    input_graph_name = "input_graph.pb"

    G2PModel, hparams = import_model_type(model_type)
    with open('%s/hparams' % model_dir, 'r') as infp:
        loaded = json.load(infp)
        hparams.parse_json(loaded)

    with ops.Graph().as_default():
        with tf.Session() as sess:
            model = G2PModel(hparams, is_training=False, with_target=False, reuse=False)
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_path) 

            #graph = sess.graph
            #print([node.name for node in graph.as_graph_def().node])


            saver = saver_lib.Saver()
            checkpoint_path = saver.save(
                sess,
                checkpoint_prefix,
                global_step=0,
                latest_filename=checkpoint_state_name)
            graph_io.write_graph(sess.graph, freeze_dir, input_graph_name)

    input_graph_path = os.path.join(freeze_dir, input_graph_name)
    input_saver_def_path = ""
    input_binary = False
    output_node_names = 'g2p/predicted_1best,g2p/probs'
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_graph_path = os.path.join(freeze_dir, output_graph_name)
    clear_devices = False

    freeze_graph.freeze_graph(
          input_graph_path, input_saver_def_path, input_binary, checkpoint_path,
          output_node_names, restore_op_name, filename_tensor_name,
          output_graph_path, clear_devices, "")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='Freezes trained model')
    arg_parser.add_argument('--model', required=True,
                            help='Path to model to freeze (dir/g2p-step)')
    arg_parser.add_argument('--freeze-dir', required=True,
                            help='Directory to put freeze artifacts')
    arg_parser.add_argument('--frozen-name', required=True,
                            help='Name of the frozen graph')
    args = arg_parser.parse_args()

    freeze_model(args.model, args.freeze_dir, args.frozen_name)
