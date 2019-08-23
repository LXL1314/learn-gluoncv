from __future__ import division
import numpy as np, time, math
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, TrainingHistory
from gluoncv.data import transforms as gcv_transforms

ctx = mx.gpu()
net = get_model('cifar_resnet20_v1', classes=10)
net.initialize(init=mx.init.Xavier(), ctx=ctx)

# data augment
transforms_train = transforms.Compose([
    gcv_transforms.RandomCrop(32, pad=4),
    transforms.RandomFlipLeftRight(),
    transforms.RandomFlipTopBottom(),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

# data loader
per_device_batch_size = 64
num_workers = 4
batch_size = per_device_batch_size * num_workers
train_loader = gluon.data.DataLoader(gluon.data.vision.CIFAR10(train=True).transform_first(transforms_train),
                                     batch_size,
                                     shuffle=True,
                                     last_batch='discard',
                                     num_workers=num_workers)
test_loader = gluon.data.DataLoader(gluon.data.vision.CIFAR10(train=False).transform_first(transforms_test),
                                    batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)

# optimizer , loss, metric (metric 是什么， 作用是什么)
lr_decay = 0.1
lr_decay_epoch = 80
trainer = gluon.Trainer(net.collect_params(), 'nag',
                        {'learning_rate': 0.1,
                         'wd': 0.0001, 'momentum': 0.9})
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
train_metric = mx.metric.Accuracy()
train_history = TrainingHistory(['training error', 'validation-error'])

# validation : acc
def test(ctx, val_data):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = batch[0].as_in_context(ctx)
        labels = batch[1].as_in_context(ctx)
        preds = net(data)
        metric.update(labels, preds)
    return metric.get()

# training
num_epoches = 160
for epoch in range(num_epoches):
    start = time.time()
    train_metric.reset()
    train_loss = 0.0
    n = 0

    if (epoch + 1) % lr_decay_epoch == 0:
        trainer.set_learning_rate(trainer.learning_rate * lr_decay)

    for i, batch in enumerate(train_loader):
        data = batch[0].as_in_context(ctx)
        labels = batch[1].as_in_context(ctx)
        with autograd.record():
            preds = net(data)
            lo = loss_fn(preds, labels).sum()
        lo.backward()
        trainer.step(batch_size)
        train_metric.update(labels, preds)
        train_loss += lo.asscalar()
        n += labels.size

    train_name, train_acc = train_metric.get()
    val_name, val_acc = test(ctx, test_loader)

    train_history.update([1 - train_acc, 1 - val_acc])

    print('[epoch %d lr %.4f] train acc=%f  loss=%f | val acc=%f | time:%.2f sec' %
          (epoch + 1, trainer.learning_rate, train_acc, train_loss / n, val_acc, time.time() - start))

train_history.plot()



