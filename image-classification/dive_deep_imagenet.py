import mxnet as mx, numpy as np, time
from mxnet import gluon, init, nd
from mxnet import autograd
from mxnet.gluon import nn
from gluoncv.utils import makedirs, TrainingHistory
from gluoncv.model_zoo import get_model

#num_gpus = 3
#ctx = [mx.gpu(i) for i in range(num_gpus)]
ctx = mx.gpu()
net = get_model('ResNet50_v2', classes=1000)
net.initialize(init.MSRAPrelu(), ctx=ctx)

jitter_param = 0.4
brightness = 0.1

