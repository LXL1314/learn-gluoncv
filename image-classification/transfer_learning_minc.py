import mxnet as mx, numpy as np, os, time, shutil
from mxnet import gluon, image, init, nd
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from gluoncv.utils import makedirs
from gluoncv.model_zoo import get_model

# set some hyperparameters
classes = 23
epoches = 120
lr = 0.001
momentum = 0.9
wd = 0.0001

lr_factor = 0.75

ctx = mx.gpu()
batch_size = 64

# data augment
jitter_param = 0.4
brightness = .1
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomFlipLeftRight(),
    transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param, saturation=jitter_param),
    transforms.RandomBrightness(brightness=brightness),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# data loader
path = '../data/minc-2500'
train_path = os.path.join(path, 'train')
val_path = os.path.join(path, 'val')
test_path = os.path.join(path, 'test')
train_data = gluon.data.DataLoader(gluon.data.vision.ImageFolderDataset(train_path).transform_first(transform_train),
                                   batch_size,
                                   shuffle=True)
val_data = gluon.data.DataLoader(gluon.data.vision.ImageFolderDataset(val_path).transform_first(transform_train),
                                 batch_size,
                                 shuffle=True)
test_data = gluon.data.DataLoader(gluon.data.vision.ImageFolderDataset(test_path),
                                  batch_size,
                                  shuffle=False)

# model, trainer, loss
finetune_net = get_model('ResNet50_v2', pretrained=True)
with finetune_net.name_scope():
    finetune_net.output = nn.Dense(classes)
finetune_net.output.initialize(init.Xavier(), ctx=ctx)
finetune_net.collect_params().reset_ctx(ctx)
finetune_net.hybridize()

trainer = gluon.Trainer(finetune_net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': momentum, 'wd': wd})
train_metric = mx.metric.Accuracy()
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

# validation and test: acc
def test(net, dataset, ctx):
    metric = mx.metric.Accuracy()
    for batch in dataset:
        data = batch[0].as_in_context(ctx)
        labels = batch[1].as_in_context(ctx)
        preds = net(data)
        metric.update(labels, preds)
    return metric.get()

# training
for epoch in range(epoches):
    start, n, train_loss = time.time(), 0, 0.0
    train_metric.reset()

    if (epoch + 1) % 50 == 0:
        trainer.set_learning_rate(trainer.learning_rate * lr_factor)

    for train_batch in train_data:
        data = train_batch[0].as_in_context(ctx)
        labels = train_batch[1].as_in_context(ctx)

        with autograd.record():
            preds = finetune_net(data)
            lo = loss_fn(preds, labels).sum()

        lo.backward()
        trainer.step(batch_size)
        train_metric.update(labels, preds)
        train_loss += lo.asscalar()
        n += labels.size

    _, train_acc = train_metric.get()
    _, val_acc = test(finetune_net, val_data, ctx)

    print('[epoch %d lr %.4f] train acc=%f  loss=%f | val acc=%f | time:%.2f sec' %
          (epoch + 1, trainer.learning_rate, train_acc, train_loss / n, val_acc, time.time() - start))










