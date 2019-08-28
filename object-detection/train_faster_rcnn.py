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
from gluoncv.model_zoo import get_model
from gluoncv.data import batchify
from gluoncv.data.transforms.presets.rcnn import FasterRCNNDefaultTrainTransform, FasterRCNNDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric

from gluoncv.utils.parallel import Parallelizable, Parallel
from gluoncv.utils.metrics.rcnn import RPNAccMetric, RPNL1LossMetric, RCNNAccMetric, \
    RCNNL1LossMetric

try:
    import horovod.mxnet as hvd
except ImportError:
    hvd = None

# mixup 是一种数据增广方式
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
    # return to x, y, im_scale
    val_batchify_fn = batchify.Tuple(*[batchify.Append() for _ in range(3)])
    val_loader = mx.gluon.data.DataLoader(val_dataset.transform(val_transform), batch_size, shuffle=False,
                            batchify_fn=val_batchify_fn, last_batch='keep')
    """
    train_loader：
    每个batch为（[data1, data2,...], [label, label2,...], [rpn_cls_targets1, rpn_cls_targets2, ...],
    [rpn_box_targets1, rpn_box_targets2, ...], [rpn_box_masks1, rpn_box_masks2, ...]）
    
    for data, label, rpn_cls_targets, rpn_box_targets, rpn_box_masks in zip(*train_loader):
        rpn_cls_targets: (1, N)   rpn_box_targets: (1, N, 4)   rpn_box_masks: (1, N, 4)
    
    val_loader:
    每个batch为（[data1, data2,...], [label, label2,...], [im_scale1, im_scale2, ...]）
    for data, label, im_scale in zip(*val_loader):
        im_scale: (1, 1) 
        但是还没搞清楚im_scale是什么东西 T_T
    
    cls_preds, box_preds, roi, samples, matches, rpn_score, rpn_box, anchors = net(data, gt_box)
    cls_targets, box_targets, box_masks = net.target_generator(roi, samples, matches, gt_label, gt_box)
            
    data:  (1, 3, h, w)
    label: (1, num_obj, 6)
            
    # rpn
    roi: (1, 128, 4)
    samples: (1, 128)
    matches: (1, 128)
            
    rpn_cls_targets: (1, N)                  rpn_score:  (1, N, 1)
    rpn_box_targets: (1, N, 4)               rpn_box:  (1, N, 4)
    rpn_box_masks:   (1, N, 4)
    
    # rcnn
    cls_targets:   (1, 128)                   cls_preds: (1, 128, num_cls + 1)
    box_targets:   (1, 128, num_cls, 4)       box_preds: (1, 128, num_cls, 4)
    rcnn box mask: (1, 128, num_cls, 4)
    """
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
    """split data to 1 batch each device , 也就是让每个device上有一个图片数据"""
    new_batch = []
    for i, data in enumerate(batch):
        new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        # val_loader中每个batch为（[data1, data2,...], [label1, label2,...], [im_scale1, im_scale2, ...]）
        # 若ctx_list长度 和 data 长度的最小值为n
        # 则 new_batch = [[data1, ..., data_n], [label1, ..., label_n], [im_scale1, ..., im_scale_n]]
        # 且每个device上有一个图片数据，即例如: data1, label1, im_scale1都在device0上面
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
            # ids:  (1, num_box, 1)
            # scores:  (1, num_box, 1)
            # bboxes:  (1, num_box, 4)
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
    return eval_metric.get()


def get_lr_at_iter(alpha): # 这个函数什么作用
    return 1. / 3. * (1 - alpha) + alpha


class ForwardBackwardTask(Parallelizable):
    def __init__(self, net, optimizer, rpn_cls_loss, rpn_box_loss,
                 rcnn_cls_loss, rcnn_box_loss, mix_ratio):
        super(ForwardBackwardTask, self).__init__()
        self.net = net
        self.optimizer = optimizer
        self.rpn_cls_loss = rpn_cls_loss
        self.rpn_box_loss = rpn_box_loss
        self.rcnn_cls_loss = rcnn_cls_loss
        self.rcnn_box_loss = rcnn_box_loss
        self.mix_ratio = mix_ratio

    def forward(self, x):
        data, label, rpn_cls_targets, rpn_box_targets, rpn_box_masks = x
        with autograd.record():
            gt_box = label[:, :, :4]
            gt_label = label[:, :, 4:5]
            cls_preds, box_preds, roi, samples, matches, rpn_score, rpn_box, anchors = self.net(data, gt_box)
            cls_targets, box_targets, box_masks = self.net.target_generator(roi, samples,
                                                                            matches, gt_label, gt_box)
            """
            data:  (1, 3, h, w)
            label: (1, num_obj, 6)
            
            # rpn
            roi: (1, 128, 4)
            samples: (1, 128)
            matches: (1, 128)
            
            rpn_cls_targets: (1, N)                  rpn_score:  (1, N, 1)
            rpn_box_targets: (1, N, 4)               rpn_box:  (1, N, 4)
            rpn_box_masks:   (1, N, 4)
    
            # rcnn
            cls_targets:   (1, 128)                   cls_preds: (1, 128, num_cls + 1)
            box_targets:   (1, 128, num_cls, 4)       box_preds: (1, 128, num_cls, 4)
            rcnn box mask: (1, 128, num_cls, 4)
            """

            # loss of rpn
            rpn_score = rpn_score.squeeze(axis=-1)
            num_rpn_pos = (rpn_cls_targets >= 0).sum()
            rpn_loss_cls = self.rpn_cls_loss(rpn_score, rpn_cls_targets, rpn_cls_targets >= 0) * \
                           rpn_cls_targets.size / num_rpn_pos
            rpn_loss_box = self.rpn_box_loss(rpn_box, rpn_box_targets, rpn_box_masks) * \
                           rpn_box_targets.size / num_rpn_pos
            rpn_loss = rpn_loss_cls + rpn_loss_box

            # loss of rcnn
            num_rcnn_pos = (cls_targets >= 0).sum()
            rcnn_loss_cls = self.rcnn_cls_loss(cls_preds, cls_targets, cls_targets >= 0) * \
                            cls_targets.size / cls_targets.shape[0] / num_rcnn_pos
            rcnn_loss_box = self.rcnn_box_loss(box_preds, box_targets, box_masks) * \
                            box_targets.size / box_targets.shape[0] / num_rcnn_pos
            rcnn_loss = rcnn_loss_cls + rcnn_loss_box

            # overall loss
            total_loss = rpn_loss.sum() * self.mix_ratio + rcnn_loss.sum() * self.mix_ratio

            rpn_cls_loss_metric = rpn_loss_cls.sum() * self.mix_ratio
            rpn_box_loss_metric = rpn_loss_box.sum() * self.mix_ratio
            rcnn_cls_loss_metric = rcnn_loss_cls.sum() * self.mix_ratio
            rcnn_box_loss_metric = rcnn_loss_box.sum() * self.mix_ratio

            rpn_acc_metric = [[rpn_cls_targets, rpn_cls_targets >= 0], [rpn_score]]
            rpn_l1_loss_metric = [[rpn_box_targets, rpn_box_masks], [rpn_box]]
            rcnn_acc_metric = [[cls_targets], [cls_preds]]
            rcnn_l1_loss_metric = [[box_targets, box_masks], [box_preds]]

        if args.amp:
            with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()

        return rpn_cls_loss_metric, rpn_box_loss_metric, rcnn_cls_loss_metric, rcnn_box_loss_metric, \
               rpn_acc_metric, rpn_l1_loss_metric, rcnn_acc_metric, rcnn_l1_loss_metric


def train(net, train_data, val_data, eval_metric, ctx, args):
    net.collect_params.reset_ctx(ctx)
    kv = mx.kvstore.create(args.kv_store)
    net.collect_params().setattr('grad_req', 'null')
    net.collect_train_params.setattr('grad_req', 'write')
    if args.horovod:
        hvd.broadcast_parameters(net.collect_params(), root_rank=0)
        trainer = hvd.DistributedTrainer(net.collect_train_params(), 'sgd',
                                         {'learning_rate': args.lr, 'wd': args.wd, 'momentum': args.momentum})
    else:
        trainer = gluon.Trainer(net.collect_train_params(), 'sgd',
                                {'learning_rate': args.lr, 'wd': args.wd, 'momentum': args.momentum},
                                update_on_kvstore=(False if args.amp else None), kvstore=kv)
    if args.amp:
        amp.init_trainer(trainer)

    lr_decay = float(args.lr_decay)
    lr_steps = sorted([float(ls) for ls in args.lr_decay_epoch.split(',') if ls.strip()])
    lr_warmup = float(args.lr_warmup)

    # losses, 以下4个loss是rcnn_task 里面要用到
    rpn_cls_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    rpn_box_loss = mx.gluon.loss.HuberLoss(rho=1 / 9.)
    rcnn_cls_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    rcnn_box_loss = mx.gluon.loss.HuberLoss()

    metrics = [mx.metric.Loss('RPN_Conf'),
               mx.metric.Loss('RPN_SmoothL1'),
               mx.metric.Loss('RCNN_CrossEntropy'),
               mx.metric.Loss('RCNN_SmoothL1')]
    # metrics: [rpn_cls_loss, rpn_box_loss, rcnn_cls_loss, rcnn_box_loss]
    # metric_losses: [[rpn_cls_loss], [rpn_box_loss], [rcnn_cls_loss], [rcnn_box_loss]]
    # metric.update(0, record)

    rpn_acc_metric = RPNAccMetric()
    rpn_bbox_metric = RPNL1LossMetric()
    rcnn_acc_metric = RCNNAccMetric()
    rcnn_bbox_metric = RCNNL1LossMetric()
    metrics2 = [rpn_acc_metric, rpn_bbox_metric, rcnn_acc_metric, rcnn_bbox_metric]

    # logger set_up
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    if args.verbose:
        logger.info('Trainable parameters:')
        logger.info(net.collect_train_params().keys())
    logger.info("Start training from [Epoch {}]".format(args.start_epoch))
    best_map = [0]

    for epoch in range(args.start_epoch, args.epochs):
        mix_ratio = 1.0
        if not args.disable_hybridization:
            net.hybridize(static_alloc=args.static_alloc)

        rcnn_task = ForwardBackwardTask(net, trainer, rpn_cls_loss, rpn_box_loss,
                                        rcnn_cls_loss, rcnn_box_loss, mix_ratio=1.0)
        executor = Parallel(1 if args.horovod else args.executor_threads, rcnn_task)
        # executor 这一句什么意思

        if args.mixup:
            train_data._dataset._data.set_mixup(np.random.uniform, 0.5, 0.5)
            mix_ratio =0.5
            if epoch >= args.epochs - args.no_mixup_epochs:
                train_data._dataset._data.set_mixup(None)
                mix_ratio = 1.0

        # 调整学习率
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info('[Epoch {}] Set learning rate to {}'.format(epoch, new_lr))

        for metric in metrics:
            metric.reset()

        tic = time.time()  # 记录一次循环的时间
        btic = time.time()  # 记录每一个batch的时间
        base_lr = trainer.learning_rate
        rcnn_task.mix_ratio = mix_ratio

        for i, batch in enumerate(train_data):
            if epoch == 0 and i <= lr_warmup:
                new_lr = base_lr * get_lr_at_iter(i / lr_warmup)
                if new_lr != trainer.learning_rate:
                    if i % args.log_interval == 0:
                        logger.info('[Epoch 0 Iteration {}] Set learning rate to {}'.format(i, new_lr))
                    trainer.set_learning_rate(new_lr)
            batch = split_and_load(batch, ctx_list=ctx)
            batch_size = len(batch[0])
            metric_losses = [[] for _ in metrics]  # metrics: [rpn_cls_loss, rpn_box_loss, rcnn_cls_loss, rcnn_box_loss]
            add_losses = [[] for _ in metrics2]  # metrics2 : [rpn_acc_metric, rpn_bbox_metric, rcnn_acc_metric, rcnn_bbox_metric]

            for data in zip(*batch):
                executor.put(data)

            for j in range(len(ctx)):
                result = executor.get()
                if (not args.horovod) or hvd.rank() == 0:
                    for k in range(len(metric_losses)):
                        metric_losses[k].append(result[k])
                    for k in range(len(add_losses)):
                        add_losses[k].append(result[len(metric_losses) + k])

            for metric, record in zip(metrics, metric_losses):
                # metrics: [rpn_cls_loss, rpn_box_loss, rcnn_cls_loss, rcnn_box_loss]
                # metric_losses: [[rpn_cls_loss], [rpn_box_loss], [rcnn_cls_loss], [rcnn_box_loss]]
                metric.update(0, record)
            for metric, records in zip(metrics2, add_losses):
            # metrics2 = [rpn_acc_metric, rpn_bbox_metric, rcnn_acc_metric, rcnn_bbox_metric]
            # add_losses: [[rpn_acc_metric], [rpn_bbox_metric], [rcnn_acc_metric], [rcnn_bbox_metric]]
            # rpn_acc_metric: [[rpn_label, rpn_weight], [rpn_cls_logits]]
                for pred in records:
                    # update(label, preds)
                    # label: [rpn_label, rpn_weight]
                    # preds: [rpn_cls_logits]
                    metric.update(pred[0], pred[1])
            trainer.step(batch_size)

            if (not args.horovod or hvd.rank() == 0) and args.log_interval and not (i + 1) % args.log_interval:
                msg = ','.join(['{} ={:.3f}'.format(*metric.get()) for metric in metrics + metrics2])
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}'.format(
                    epoch, i, args.batch_size * args.log_interval / (time.time() - btic), msg))
                btic = time.time()

        if (not args.horovod) or hvd.rank() == 0:
            msg = ','.join(['{} ={:.3f}'.format(*metric.get()) for metric in metrics])
            logger.info('[Epoch {}] Training cost: {:.3f}, {}'.format(
                epoch, (time.time() - tic), msg))
            if (epoch % args.val_interval == 0) or (args.save_interval and epoch % args.save_interval == 0):
                # 每循环args.val_interval或者args.save_interval次
                # 就需要使用验证集来测试一次，得到current_map
                map_name, mean_ap = validate(net, val_data, ctx, eval_metric, args)
                val_msg = "\n".join('{}={}'.format(k, v) for k, v in zip(map_name, mean_ap))
                logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
                current_map = float(mean_ap[-1])  # mean_ap的最后一个数据就是mAP
            else:
                current_map = 0
            save_params(net, logger, best_map, current_map, epoch, args.save_interval, args.save_prefix)
        executor.__del__()


if __name__ == '__main__':
    import sys
    sys.setrecursionlimit(1100)
    args = parse_args()

    if args.amp:
        amp.init()

    # ctx
    if args.horovod:
        ctx = [mx.gpu(hvd.local_rank())]
        args.batch_size = hvd.size()
    else:
        ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
        ctx = ctx if ctx else [mx.cpu()]
        args.batch_size = len(ctx)

    # network
    kwargs = {}
    module_list = []
    if args.use_fpn:
        module_list.append('fpn')
    if args.norm_layer is None:
        module_list.append(args.norm_layer)
        if args.norm_layer == 'bn':
            kwargs['num_devices'] = len(args.gpus.split(','))
    net_name = '_'.join(('faster_rcnn', *module_list, args.network, args.dataset))
    args.save_prefix += net_name
    net = get_model(net_name, pretrained=True, **kwargs)
    if args.resume.strip():
        net.load_parameters(args.resume.strip())
    else:
        for param in net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize()

    # get data
    train_dataset, val_dataset, eval_metric = get_dataset(args.dataset, args)
    batch_size = 1 if args.horovod else args.batch_size
    train_data, val_data = get_loader(net, train_dataset, val_dataset, batch_size, args)

    # training
    train(net, train_data, val_data, eval_metric, ctx, args)