import os
import argparse

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import logging
import time
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.contrib import amp
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data import batchify
from gluoncv.data.transforms.presets.rcnn import FasterRCNNDefaultTrainTransform, FasterRCNNDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric

# 下面这两个是需要自己实现吗？？？
from gluoncv.utils.parallel import Parallelizable, Parallel
from gluoncv.utils.metrics.rcnn import RPNAccMetric, RPNL1LossMetric, RCNNAccMetric, \
    RCNNL1LossMetric

try:
    import horovod.mxnet as hvd
except ImportError:
    hvd = None

def parse_args():
    parser = argparse.ArgumentParser(description="Train Faster-RCNN networks end to end")
    parser.add_argument("--network", type='str', default="resnet50_v1b",
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset. Now support voc and coco.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers, you can use larger '
                                        'number to accelerate data loading, '
                                        'if your CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=str, default='',
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                             'For example, you can resume from ./faster_rcnn_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                             'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=str, default='',
                        help='Learning rate, default is 0.001 for voc single gpu training.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='',
                        help='epochs at which learning rate decays. default is 14,20 for voc.')
    parser.add_argument('--lr-warmup', type=int, default=0,
                        help='warmup iterations to adjust learning rate, default is 0 for voc.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=str, default='',
                        help='Weight decay, default is 5e-4 for voc')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=1,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                       help='Print helpful debugging info once set.')
    parser.add_argument('--mixup', action='store_true', help='Use mixup training.')
    parser.add_argument('--no-mixup-epochs', type=int, default=20,
                        help='Disable mixup training if enabled in the last N epochs.')
    # Norm layer options
    parser.add_argument('--norm-layer', type=str, default=None,
                        help='Type of normalization layer to use. '
                             'If set to None, backbone normalization layer will be fixed,'
                             ' and no normalization layer will be used. '
                             'Currently supports \'bn\', and None, default is None')
    # FPN options
    parser.add_argument('--use-fpn', action='store_true',
                        help='Whether to use feature pyramid network.')
    # Performance options
    parser.add_argument('--disable-hybridization', action='store_true',
                        help='Whether to disable hybridize the model. '
                             'Memory usage and speed will decrese.')
    parser.add_argument('--static-alloc', action='store_true',
                        help='Whether to use static memory allocation. Memory usage will increase.')
    parser.add_argument('--amp', action='store_true',
                        help='Use MXNet AMP for mixed precision training.')
    parser.add_argument('--horovod', action='store_true',
                        help='Use MXNet Horovod for distributed training. Must be run with OpenMPI. '
                             '--gpus is ignored when using --horovod.')
    parser.add_argument('--executor-threads', type=int, default=1,
                        help='Number of threads for executor for scheduling ops. '
                             'More threads may incur higher GPU memory footprint, '
                             'but may speed up throughput.')
    parser.add_argument('--kv-store', type=str, default='nccl',
                        help='KV store options. local, device, nccl, dist_sync, dist_device_sync, '
                             'dist_async are available.')

    args = parse_args()

    if args.hvd:
        if hvd is None:
            raise SystemExit("Horovod is not found, please check if you installed it correctly")
        hvd.init()

    if args.dataset == 'voc':
        args.epochs = int(args.epochs) if args.epochs else 20
        args.lr_decay_epoch = args.lr_decay_epoch if args.lr_decay_epoch else '14, 20'
        args.lr = float(args.lr) if args.lr else 0.001
        args.lr_warmup = args.lr_warmup if args.lr_warmup else -1
        args.wd = float(args.wd) if args.wd else 5e-4
    elif args.dataset == 'coco':
        args.epochs = int(args.epochs) if args.epochs else 26
        args.lr_decay_epoch = args.lr_decay_epoch if args.lr_decay_epoch else '17,23'
        args.lr = float(args.lr) if args.lr else 0.01
        args.lr_warmup = args.lr_warmup if args.lr_warmup else 1000
        args.wd = float(args.wd) if args.wd else 1e-4

    return args


def get_dataset(args):
    dataset = args.dataset
    if dataset.lower() == 'voc':
        train_dataset = gdata.VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])
        val_dataset = gdata.VOCDetection(splits=[(2007, 'test')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower == 'coco':
        train_dataset = gdata.COCODetection(splits='instances_train2017', use_crowd=True)
        val_dataset = gdata.COCODetection(splits='instances_val2017', skip_empty=False)
        val_metric = COCODetectionMetric(val_dataset, args.save_prefix + '_eval', cleanup=True)
    else:
        raise NotImplementedError("dataset: {} not implemented".format(dataset))

    if args.mixup:
        from gluoncv.data.mixup import detection
        train_dataset = detection.MixupDetection(train_dataset)

    return train_dataset, val_dataset, val_metric


def get_loader(net, train_dataset, val_dataset, batch_size, args):
    train_transform = FasterRCNNDefaultTrainTransform(net.short, net.max_size, net,
                                                      ashape=net.ashape, multi_stage=args.use_fpn)
    # return images, labels, rpn_cls_targets, rpn_box_targets, rpn_box_masks
    train_batchify_fn = batchify.Tuple(*[batchify.Append() for _ in range(5)])
    train_loader = mx.gluon.data.DataLoader(train_dataset.transform(train_transform),
        batch_size, shuffle=True, batchify_fn=train_batchify_fn, last_batch='rollover')

    short = net.short[-1] if isinstance(net.short, (tuple, list)) else net.short
    val_transform = FasterRCNNDefaultValTransform(short, net.max_size)
    # return to what? why 3？
    val_batchify_fn = batchify.Tuple(*[batchify.Append() for _ in range(3)])
    val_loader = mx.gluon.data.DataLoader(val_dataset.transform(val_transform), batch_size, shuffle=False,
                            batchify_fn=val_batchify_fn, last_batch='keep')
    return train_loader, val_loader


def save_params(net, logger, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        logger.info("[Epoch {}] current mAP {} higher than current best mAP {}, saving to {}".format(
            epoch, current_map, best_map, '{:s}_best.params'.format(prefix)))
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(prefix))
        with open(prefix + '_best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and (epoch + 1) % save_interval == 0:
        logger.info('[Epoch {}] Saving parameters to {}'.format(
            epoch, '{:s}_{:.04d}_{:.4f}.params'.format(prefix, epoch, current_map)))
        net.save_parameters('{:s}_{:.04d}_{:.4f}.params'.format(prefix, epoch, current_map))


def split_and_load(batch, ctx_list):
    """split data to 1 batch each device 什么意思？_？"""
    new_batch = []
    for i, data in enumerate(batch):
        new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        new_batch.append(new_data)
    return new_batch


def validate(net, val_data, ctx, eval_metric, args):
    """在验证集上测试当前训练的网络"""
    clipper = gcv.nn.bbox.BBoxClipToImage()
    eval_metric.reset()
    if not args.disable_hybridization:
        net.hybridize(static_alloc=args.static_alloc)
    for batch in val_data:
        batch = split_and_load(batch, ctx_list=ctx)
        det_bboxes, det_ids, det_scores = [], [], []
        gt_bboxes, gt_ids, gt_difficults = [], [], []
        for x, y, im_scale in zip(*batch):
            ids, scores, bboxes = net(x)

            det_ids.append(ids)
            det_scores.append(scores)
            det_bboxes.append(clipper(bboxes, x))
            im_scale = im_scale.reshape((-1)).asscalar()
            det_bboxes[-1] *= im_scale

            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_bboxes[-1] *= im_scale
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
        """
    for det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff in zip(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults):
        eval_metric.update(det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff)
        """
    return eval_metric.get()


def get_lr_at_iter(alpha): # 这个函数什么作用
    return 1. / 3. * (1 - alpha) + alpha


class ForwardBackwardTask(Parallelizable):
    def __init__(self, ):
        super(ForwardBackwardTask, self).__init__()






