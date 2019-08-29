import argparse
import os
import logging
import time
import warnings
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultTrainTransform
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.data.dataloader import RandomTransformDataLoader
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.utils import LRScheduler, LRSequential


def parse_args():
    parser = argparse.ArgumentParser("Train YOLO networks with random input shape")
    parser.add_argument('--network', type=str, default='darknet53',
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--data-shape', type=int, default=416,
                        help="Input data shape for evaluation, use 320, 416, 608... " +
                             "Training is with random shapes from (320 to 608).")
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset. Now support voc.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers, you can use larger '
                                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                             'For example, you can resume from ./yolo3_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                             'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate, default is 0.001')
    parser.add_argument('--lr-mode', type=str, default='step',
                        help='learning rate scheduler mode. options are step, poly and cosine.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='interval for periodic learning rate decays. default is 0 to disable.')
    parser.add_argument('--lr-decay-epoch', type=str, default='160,180',
                        help='epochs at which learning rate decays. default is 160,180.')
    parser.add_argument('--warmup-lr', type=float, default=0.0,
                        help='starting warmup learning rate. default is 0.0.')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='number of warmup epochs.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 5e-4')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--num-samples', type=int, default=-1,
                        help='Training images. Use -1 to automatically get the number.')
    parser.add_argument('--syncbn', action='store_true',
                        help='Use synchronize BN across devices.')
    parser.add_argument('--no-random-shape', action='store_true',
                        help='Use fixed size(data-shape) throughout the training, which will be faster '
                             'and require less memory. However, final model will be slightly worse.')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
    parser.add_argument('--mixup', action='store_true',
                        help='whether to enable mixup.')
    parser.add_argument('--no-mixup-epochs', type=int, default=20,
                        help='Disable mixup training if enabled in the last N epochs.')
    parser.add_argument('--label-smooth', action='store_true', help='Use label smoothing.')
    args = parser.parse_args()
    return args


def get_dataset(args):
    dataset = args.dataset
    if dataset.lower() == 'voc':
        train_dataset = gdata.VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])
        val_dataset = gdata.VOCDetection(splits=[(2007, 'test')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))

    if args.num_samples < 0:
        args.num_samples = len(train_dataset)

    if args.mixup:
        from gluoncv.data.mixup import detection
        train_dataset = detection.MixupDetection(train_dataset)
    return train_dataset, val_dataset, val_metric


def get_loader(net, train_dataset, val_dataset, data_shape, batch_size, num_workers, args):
    width, height = data_shape, data_shape
    batchify_fn = Tuple(*([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(1)]))
    if args.no_random_shape:
        train_transform = YOLO3DefaultTrainTransform(width, height, net, mixup=args.mixup)
        # return (img, objectness[0], center_targets[0], scale_targets[0], weights[0],
        #                 class_targets[0], gt_bboxes[0])
        train_loader = gluon.data.DataLoader(train_dataset.transform(train_transform),
                                             batch_size, shuffle=True, last_batch='rollover',
                                             batchify_fn=batchify_fn, num_workers=num_workers)
    else:
        transform_fns = [YOLO3DefaultTrainTransform(x * 32, x * 32, net, mixup=args.mixup) for x in range(10, 20)]
        train_loader = RandomTransformDataLoader(transform_fns, dataset=train_dataset, batch_size=batch_size,
                                                 shuffle=True, interval=10, last_batch='rollover',
                                                 batchify_fn=batchify_fn, num_workers=num_workers)

    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))  # stack image, and pad labels： labels的0-4：box, 4-5: cls
    val_transform = YOLO3DefaultValTransform(width, height)
    # return img, bbox.astype(img.dtype)
    val_loader = gluon.data.DataLoader(val_dataset.transform(val_transform), batch_size, False,
                                       last_batch='keep', batchify_fn=val_batchify_fn, num_workers=num_workers)
    return train_loader, val_loader


def save_parameters(net, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_parameters("{:s}_best.params".format(prefix, epoch, current_map))
        with open(prefix + "_best_map.log", 'a') as f:
            f.write("{:.4f}:\t{:.4f}\n".format(epoch, current_map))
    if save_interval and epoch % save_interval == 0:
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))


def validate(net, val_data, ctx, eval_metric):
    eval_metric.reset()
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    mx.nd.waitall()
    net.hybridize()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        det_ids, det_scores, det_bboxes= [], [], []
        gt_ids, gt_difficulties, gt_bboxes  = [], [], []
        for x, y in zip(data, label):
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            det_bboxes.append(bboxes.clip(0, x.shape[1]))

            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_difficulties.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficulties)
    return eval_metric.get()


def train(net, train_data, val_data, eval_metric, ctx, args):
    net.collect_params().reset_ctx(ctx)
    if args.no_wd:
        for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
            v.wd_mult = 0.0

    if args.label_smooth:
        net._target_generator._label_smooth = True

    if args.lr_decay_period > 0:
        lr_decay_epoch = list(range(args.lr_decay_period, args.epochs, args.lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.split(',')]

    lr_decay_epoch = [e - args.warmup_epochs for e in lr_decay_epoch]
    num_batches = args.num_samples // args.batch_size

    lr_scheduler = LRSequential([
        LRScheduler('linear', base_lr=0, target_lr=args.lr, nepochs=args.warmup_epochs, iters_per_epoch=num_batches),
        LRScheduler(args.mode, base_lr=args.lr, nepochs=args.epochs - args.warmup_epochs,
                    iters_per_epoch=num_batches,
                    step_epoch=lr_decay_epoch,
                    step_factor=args.lr_decay, power=2)])

    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'wd': args.wd, 'momentum': args.momentum, 'lr_scheduler': lr_scheduler},
                            kvstore='local')

    obj_metrics = mx.metric.Loss("ObjLoss")
    center_metrics = mx.metric.Loss("BoxCenterLoss")
    scale_metrics = mx.metric.Loss("BoxScaleLoss")
    cls_metrics = mx.metric.Loss("ClassLoss")

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
    logger.info("Start training from [Epoch {}]".format(args.start_epoch))
    best_map = [0]

    for epoch in range(args.start_epoch, args.epochs):
        if args.mixup:
            try:
                train_data._dataset.set_mixup(np.random.beta, 1.5, 1.5)
            except AttributeError:
                train_data._dataset._data.set_mixup(np.random.beta, 1.5, 1.5)
            if epoch >= args.epochs - args.no_mixup_epochs:
                try:
                    train_data._dataset.set_mixup(None)
                except AttributeError:
                    train_data._dataset._data.set_mixup(None)

        tic = time.time()
        btic = time.time()
        mx.nd.waitall()
        net.hybridize()

        for i, batch in enumerate(train_data):
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            fixed_targets = [gluon.utils.split_and_load(batch[it], ctx_list=ctx, batch_axis=0) for it in range(1, 6)]
            # objectness, center_targets, scale_targets, weights, class_targets
            # objectness, center_targets, scale_targets 代表什么？
            gt_boxes = gluon.utils.split_and_load(batch[6], ctx_list=ctx, batch_axis=0)


            with autograd.record():
                obj_losses, center_losses, scale_losses, cls_losses = net(data, gt_boxes, *[ft for ft in fixed_targets])
                sum_losses = obj_losses + center_losses + scale_losses + cls_losses
                sum_losses.backward()

            trainer.step(batch_size)
            obj_losses.update(0, obj_losses)
            center_metrics.update(0, center_losses)
            scale_metrics.update(0, scale_losses)
            cls_metrics.update(0, cls_losses)

            if args.log_interval and (i + 1) % args.log_interval == 0:
                name1, loss1 = obj_metrics.get()
                name2, loss2 = center_metrics.get()
                name3, loss3 = scale_metrics.get()
                name4, loss4 = scale_metrics.get()
                logger.info('[Epoch {}][Batch {}], LR: {:.2E}, Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}, '
                            '{}={:.3f}, {}={:.3f}'.format(epoch, i, trainer.learning_rate,
                                                          batch_size / (time.time() - btic), name1, loss1, name2, loss2,
                                                          name3, loss3, name4, loss4))
            btic = time.time()

        name1, loss1 = obj_metrics.get()
        name2, loss2 = center_metrics.get()
        name3, loss3 = scale_metrics.get()
        name4, loss4 = scale_metrics.get()
        logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
            epoch, (time.time() - tic), name1, loss1, name2, loss2, name3, loss3, name4, loss4))

        if ((epoch + 1) % args.val_interval == 0) or (args.save_interval and epoch % args.save_interval == 0):
            map_name, mean_map = validate(net, val_data, ctx, eval_metric)
            val_msg = "\n".join('{}={}'.format(k, v) for k, v in zip(map_name, mean_map))
            logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            current_map = float(mean_map[-1])
        else:
            current_map = 0.
        save_parameters(net, best_map, current_map, epoch, args.save_interval, args.save_prefix)


if __name__ == '__main__':
    args = parse_args()
    gutils.random.seed(args.seed)

    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    net_name = '_'.join(('yolo3', args.network, args.dataset))
    args.save_prefix += net_name

    if args.syncbn and len(ctx) > 1:
        net = get_model(net_name, pretrained_base=True, norm_layer=gluon.contrib.nn.SyncBatchNorm,
                        norm_kwargs={'num_devices': len(ctx)})
        async_net = get_model(net_name, pretrained_base=False)  # used by cpu worker???
    else:
        net = get_model(net_name, pretrained_base=True)
        async_net = net

    if args.resume.strip():
        net.load_parameters(args.resume.strip())
        async_net.load_parameters(args.resume.strip())
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            net.initialize()
            async_net.initialize()

    train_dataset, val_dataset, eval_metric = get_dataset(args)
    train_data, val_data = get_loader(async_net, train_dataset, val_dataset, args.data_shape,
                                      args.batch_size, args.num_workers, args)

    train(net, train_data, val_data, eval_metric, ctx, args)