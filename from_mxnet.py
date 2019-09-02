# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
.. _tutorial-from-mxnet:

Compile MXNet Models
====================
**Author**: `Joshua Z. Zhang <https://zhreshold.github.io/>`_, \
            `Kazutaka Morita <https://github.com/kazum>`_

This article is an introductory tutorial to deploy mxnet models with Relay.

For us to begin with, mxnet module is required to be installed.

A quick solution is

.. code-block:: bash

    pip install mxnet --user

or please refer to offical installation guide.
https://mxnet.incubator.apache.org/versions/master/install/index.html
"""
# some standard imports
import mxnet as mx
import tvm
import tvm.relay as relay
import numpy as np
from tvm.contrib import util
import os

dtype = 'float32'
use_android = False
use_arm64 = False

network = 'mnet.25'
device = 'x86.cuda'

device_key = 'rk3288'
log_file = "./logs/%s.%s.%s.log" % (device_key, network, device)

path = '/home/hower/models/mxnet/%s' % (network)

#set the input shape/layer
input_layer = 'data'
batch_size = 1
if network == 'mnet.25':
    image_shape = (3, 640, 480)
elif network == 'MobileFaceNets':
    image_shape = (3, 112, 112)
else:
    image_shape = (3, 224, 224)
input_shape = (batch_size,) + image_shape

######################################################################
#set the target/target_host
if device == 'arm.cpu':
    if use_arm64:
        target = tvm.target.create('llvm -device=arm_cpu -target=arm64-linux-android -mattr=+neon')
    else:
        target = tvm.target.create('llvm -device=arm_cpu -target=arm-linux-androideabi -mattr=+neon -mfloat-abi=soft')

    #target = tvm.target.create('llvm -device=arm_cpu -target=aarch64-linux-gnu -mattr=+neon')
    target_host = None

elif device == 'arm.gpu':
    #target = tvm.target.create('opencl -device=mali')
    #target_host = 'llvm -target=aarch64-linux-gnu -mattr=+neon'
    target = tvm.target.create('opencl -device=mali')

    if use_arm64:
        target_host = 'llvm -target=arm64-linux-android -mattr=+neon'
    else:
        target_host = 'llvm -target=arm-linux-androideabi -mattr=+neon -mfloat-abi=soft'

elif device == 'x86.cpu':
    target = 'llvm'
    target_host = None

elif device == 'x86.cuda':
    target = 'cuda'
    target_host = 'llvm'

else:
    target = tvm.target.create('llvm -target=arm64-linux-android')
    target_host = None


######################################################################
# input the mxnet model
mx_sym, args, auxs = mx.model.load_checkpoint(path, 0)


######################################################################
# Compile the Graph
# -----------------
# Now we would like to port the Gluon model to a portable computational graph.
# It's as easy as several lines.
# We support MXNet static graph(symbol) and HybridBlock in mxnet.gluon
shape_dict = {input_layer: input_shape}
func, params = relay.frontend.from_mxnet(mx_sym, shape_dict, dtype, args, auxs)


######################################################################
# now compile the graph
with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(func, target, params=params)

######################################################################
#build the relay model
# compile kernels with history best records
print("Compile...")

######################################################################
#save the relay model
temp = util.tempdir()
path_lib = temp.relpath("%s.%s.so" % (path, device))

if use_android:
    from tvm.contrib import ndk
    if use_arm64:
        lib.export_library(path_lib, ndk.create_shared)
    else:
        lib.export_library(path_lib, ndk.create_shared, options=["-shared", "-fPIC", "-mfloat-abi=softfp", "-mfpu=neon"])

else:
    lib.export_library(path_lib)
    #lib.export_library(path_lib, tvm.contrib.cc.create_shared, cc="aarch64-linux-gnu-g++")

with open(temp.relpath("%s.%s.json" % (path, device)), "w") as fo:
    fo.write(graph)
with open(temp.relpath("%s.%s.params" % (path, device)), "wb") as fo:
    fo.write(relay.save_param_dict(params))


print("------convert done!!!------")


