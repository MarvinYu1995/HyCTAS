from __future__ import division
import os
import sys
import time
import glob
import json
import logging
import argparse
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from tensorboardX import SummaryWriter

import numpy as np
from thop import profile

from config_train import config

# if config.is_eval:
#     config.save = '../OUTPUT/eval-{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
# else:
#     config.save = '../OUTPUT/train-{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))

from dataloader import get_train_loader, CyclicIterator
from datasets import Cityscapes, COCO
from eval import SegEvaluator

import dataloaders
from utils.init_func import init_weight
from utils.lr_scheduler import Iter_LR_Scheduler
from seg_opr.loss_opr import ProbOhemCrossEntropy2d
from seg_opr.loss_opr import L1Loss, MSELoss, DeepLabCE, RegularCE
from utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat
from utils.dist_utils import reduce_tensor, ModelEma
from hrtlab import HRTLab
from collections import OrderedDict
from utils.pyt_utils import AverageMeter
from utils.dist_utils import reduce_tensor

import yaml
import timm
from timm.optim import create_optimizer
from utils.pyt_utils import AverageMeter, to_cuda, get_loss_info_str, compute_hist, compute_hist_np, load_pretrain

from detectron2.config import get_cfg
from detectron2.engine import launch, default_setup, default_argument_parser
import detectron2.data.transforms as T
from detectron2.structures import BitMasks, ImageList, Instances
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.projects.panoptic_deeplab import (
    PanopticDeeplabDatasetMapper,
    add_panoptic_deeplab_config,
)

## dist train
try:
    import apex
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    from torch.nn.parallel import DistributedDataParallel as DDP
    has_apex = False

def adjust_learning_rate(base_lr, power, optimizer, epoch, total_epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * power


# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='../configs/panoptic/512drop0.2.yaml', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--det2_cfg', type=str, default='configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml', help='')
parser.add_argument('--save', type=str, default='../OUTPUT/train_', help='')
parser.add_argument('--exp_name', type=str, default='3path799', help='')
parser.add_argument('--pretrain', type=str, default=None, help='resume path')
parser.add_argument('--resume', type=str, default='../OUTPUT/train/', help='resume path')
parser.add_argument('--clip-grad', type=float, default=5., metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')

parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--world_size", default=1, type=int)
parser.add_argument("--eval_height", default=1025, type=int, help='train height')
parser.add_argument("--eval_width", default=2049, type=int, help='train width')
parser.add_argument("--test_epoch", default=250, type=int, help='Epochs for test')
parser.add_argument("--batch_size", default=12, type=int, help='batch size')
parser.add_argument("--Fch", default=12, type=int, help='Fch')
parser.add_argument('--stem_head_width', type=float, default=1.0, help='base learning rate')

## new retrain ###
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--epochs', type=int, default=4000, help='num of training epochs')
parser.add_argument('--dataset', type=str, default='cityscapes', help='pascal or cityscapes')
parser.add_argument('--base_lr', type=float, default=0.05, help='base learning rate')
parser.add_argument('--warmup_start_lr', type=float, default=5e-6, help='warm up learning rate')
parser.add_argument('--lr-step', type=float, default=None)
parser.add_argument('--warmup-iters', type=int, default=1000)
parser.add_argument('--min-lr', type=float, default=None)
parser.add_argument('--crop_size', type=int, default=512, help='image crop size')
parser.add_argument('--resize', type=int, default=512, help='image crop size')
parser.add_argument("--image_height", default=513, type=int, help='train height')
parser.add_argument("--image_width", default=1025, type=int, help='train width')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--dist', type=bool, default=True)
parser.add_argument('--autodeeplab', type=str, default='train_seg')
parser.add_argument('--max-iteration', default=1000000, type=bool)
parser.add_argument('--mode', default='poly', type=str, help='how lr decline')
parser.add_argument('--train_mode', type=str, default='iter', choices=['iter', 'epoch'])

parser.add_argument("--data_path", default='/home/t-hongyuanyu/data/cityscapes', type=str, help='If specified, replace config.load_path')
parser.add_argument("--load_path", default='', type=str, help='If specified, replace config.load_path')
parser.add_argument("--json_file", default='jsons/0.json', type=str, help='model_arch')
parser.add_argument("--seed", default=12345, type=int, help="random seed")
parser.add_argument('--sync_bn', action='store_false',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--random_sample', action='store_true',
                    help='Random sample path.')
parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path prob')

# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

parser.add_argument('--eval_flag', action='store_true', default=False,
                    help='semantic eval')
# Loss
parser.add_argument('--bn_momentum', type=float, default=0.01, help='bn momentum')
parser.add_argument('--lamb', type=float, default=0.2, help='deep sup')
parser.add_argument('--ignore', type=int, default=255, help='semantic ignore')
parser.add_argument('--topk_percent', type=float, default=0.2, help='semantic topk_percent')
parser.add_argument('--semantic_loss_weight', type=float, default=1.0, help='semantic loss weight')
parser.add_argument('--center_loss_weight', type=float, default=200., help='center loss weight')
parser.add_argument('--offset_loss_weight', type=float, default=0.01, help='offset loss weight')

# train val
parser.add_argument('--norm', type=str, default='naiveSyncBN', help='BN, SyncBN, naiveSyncBN')
parser.add_argument('--sem_only', type=bool, default=False, help='sem_only')
parser.add_argument('--use_aux', type=bool, default=True, help='use_aux')
parser.add_argument('--align_corners', type=bool, default=False, help='align_corners')
parser.add_argument('--model_type', type=int, default=1, help='0 s=8, no fusion, 1, fuse s4, 2 fuse s4, s8')
parser.add_argument('--eval_flip', action='store_true', default=False,
                    help='semantic eval flip')

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
    augs.append(T.RandomFlip())
    return augs



def main():
    args, args_text = _parse_args()

    # dist init
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    config.device = 'cuda:%d' % args.local_rank
    torch.cuda.set_device(args.local_rank)
    args.world_size = torch.distributed.get_world_size()
    args.local_rank = torch.distributed.get_rank()
    logging.info("rank: {} world_size: {}".format(args.local_rank, args.world_size))

    # detectron2 data loader ###########################
    # det2_args = default_argument_parser().parse_args()
    det2_args = args
    det2_args.config_file = args.det2_cfg
    cfg = setup(det2_args)
    cfg.defrost()
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size * args.world_size
    cfg.freeze()
    mapper = PanopticDeeplabDatasetMapper(cfg, augmentations=build_sem_seg_train_aug(cfg))
    det2_dataset = iter(build_detection_train_loader(cfg, mapper=mapper))
    
    if args.load_path:
        config.load_path = args.load_path

    config.batch_size = args.batch_size
    config.image_height = args.image_height
    config.image_width = args.image_width
    config.eval_height = args.eval_height
    config.eval_width = args.eval_width
    config.Fch = args.Fch
    config.dataset_path = args.data_path
    config.save = args.save + args.exp_name

    if args.local_rank == 0:
        create_exp_dir(config.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh'))
        logger = SummaryWriter(config.save)
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
        logging.info("args = %s", str(config))
    else:
        logger = None

    # preparation ################
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
    train_loader, train_sampler, val_loader, val_sampler, num_classes = dataloaders.make_data_loader(args, **kwargs)

    with open(args.json_file, 'r') as f:
        model_dict = json.loads(f.read())

    if args.dataset == "cityscapes":
        semantic_loss = DeepLabCE(ignore_label=args.ignore, top_k_percent_pixels=args.topk_percent)
    elif args.dataset == "coco":
        semantic_loss = RegularCE(ignore_label=args.ignore)

    semantic_loss_weight = args.semantic_loss_weight
    center_loss = MSELoss(reduction='none')
    center_loss_weight = args.center_loss_weight
    offset_loss = L1Loss(reduction='none')
    offset_loss_weight = args.offset_loss_weight

    model = HRTLab(model_dict["ops"], model_dict["paths"], model_dict["downs"], model_dict["widths"], model_dict["lasts"],
        semantic_loss, semantic_loss_weight, center_loss, center_loss_weight, offset_loss, offset_loss_weight, lamb=args.lamb, eval_flag=args.eval_flag, num_classes=num_classes, layers=config.layers, Fch=config.Fch, width_mult_list=config.width_mult_list, stem_head_width=(args.stem_head_width, args.stem_head_width), norm=args.norm, align_corners=args.align_corners, pretrain=args.pretrain, model_type=args.model_type, sem_only=args.sem_only, use_aux=args.use_aux)

    last = model_dict["lasts"]

    if args.local_rank == 0:
        logging.info("net: " + str(model))
        for b in range(len(last)):
            if len(config.width_mult_list) > 1:
                plot_op(model.ops[b], model.paths[b], width=model.widths[b], head_width=args.stem_head_width, F_base=config.Fch).savefig(os.path.join(config.save, "ops_%d_%d.png"%(0,b)), bbox_inches="tight")
            else:
                plot_op(model.ops[b], model.paths[b], F_base=config.Fch).savefig(os.path.join(config.save, "ops_%d_%d.png"%(0,b)), bbox_inches="tight")
        plot_path_width(model.lasts, model.paths, model.widths).savefig(os.path.join(config.save, "path_width%d.png"%0))
        
        flops, params = profile(model, inputs=(torch.randn(1, 3, args.eval_height, args.eval_width),), verbose=False)
        logging.info("params = %fMB, FLOPs = %fGB", params / 1e6, flops / 1e9)
        logging.info("ops:" + str(model.ops))
        logging.info("path:" + str(model.paths))
        logging.info("last:" + str(model.lasts))
        with open(os.path.join(config.save, 'args.yaml'), 'w') as f:
            f.write(args_text)

    init_weight(model, nn.init.kaiming_normal_, torch.nn.BatchNorm2d, config.bn_eps, config.bn_momentum, mode='fan_in', nonlinearity='relu')


    # for m in model.modules():
    #     if isinstance(m, nn.Conv2d):
    #         nn.init.normal_(m.weight, std=0.001)
    #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
    #         nn.init.constant_(m.weight, 1)
    #         nn.init.constant_(m.bias, 0)

    # # set batchnorm momentum
    # for module in model.modules():
    #     if isinstance(module, torch.nn.BatchNorm2d):
    #         module.momentum = args.bn_momentum

    model = model.cuda()

    # Optimizer ###################################
    base_lr = args.base_lr

    if args.opt == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-08)
    elif args.opt == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    else:
        optimizer = create_optimizer(args, model)
        
    if args.sched == "raw":
        lr_scheduler = None
    else:
        max_iteration = len(train_loader) * args.epochs
        lr_scheduler = Iter_LR_Scheduler(args, max_iteration, len(train_loader))

    start_epoch = 0

    if os.path.exists(os.path.join(config.save, 'last.pth.tar')):
        args.resume = os.path.join(config.save, 'last.pth.tar')

    if args.resume:
        model_state_file = args.resume
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location=torch.device('cpu'))
            start_epoch = checkpoint['start_epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            logging.info('Loaded checkpoint (starting from iter {})'.format(checkpoint['start_epoch']))


    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=None)

    # if args.sync_bn:
    #     if has_apex:
    #         model = apex.parallel.convert_syncbn_model(model)
    #     else:
    #         model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    if has_apex:
        model = DDP(model, delay_allreduce=True)
    else:
        model = DDP(model, device_ids=[args.local_rank])

    if model_ema:
        eval_model = model_ema.ema
    else:
        eval_model = model

    best_valid_iou = -9999.
    best_epoch = 0
    temp_iou = -99999
    avg_loss = 99999

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        if args.local_rank == 0:
            logging.info(config.load_path)
            logging.info(config.save)
            logging.info("lr: " + str(optimizer.param_groups[0]['lr']))

        # training
        drop_prob = args.drop_path_prob * epoch / args.epochs
        model.module.backbone.drop_path_prob(drop_prob)

        train(train_loader, det2_dataset, model, model_ema, lr_scheduler, optimizer, logger, epoch, args, cfg)
        
        if args.dataset == 'coco':
            temp_iou, avg_loss = validation(val_loader, eval_model, semantic_loss, num_classes, args, cal_miou=True)
        else:
            # if (epoch + 1) >= args.epochs // 4:
            if (epoch + 1) >= args.epochs // 4:
                temp_iou, avg_loss = validation(val_loader, eval_model, semantic_loss, num_classes, args, cal_miou=True)
            else:
                temp_iou = 0.
                avg_loss = 99999.

        torch.cuda.empty_cache()
        if args.local_rank == 0:
            if temp_iou > best_valid_iou:
                best_valid_iou = temp_iou
                best_epoch = epoch

                if model_ema is not None:
                    torch.save({
                        'start_epoch': epoch + 1,
                        'state_dict': model_ema.ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        # 'lr_scheduler': lr_scheduler.state_dict(),
                    }, os.path.join(config.save, 'best_checkpoint.pth.tar'))
                else:
                    torch.save({
                        'start_epoch': epoch + 1,
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        # 'lr_scheduler': lr_scheduler.state_dict(),
                    }, os.path.join(config.save, 'best_checkpoint.pth.tar'))

            logger.add_scalar("mIoU/val", temp_iou, epoch)
            logging.info("[Epoch %d/%d] valid mIoU %.4f eval loss %.4f"%(epoch + 1, args.epochs, temp_iou, avg_loss))
            logging.info("Best valid mIoU %.4f Epoch %d"%(best_valid_iou, best_epoch + 1 ))

            if model_ema is not None:
                torch.save({
                    'start_epoch': epoch + 1,
                    'state_dict': model_ema.ema.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    # 'lr_scheduler': lr_scheduler.state_dict(),
                }, os.path.join(config.save, 'last.pth.tar'))
            else:
                torch.save({
                        'start_epoch': epoch + 1,
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        # 'lr_scheduler': lr_scheduler.state_dict(),
                    }, os.path.join(config.save, 'last.pth.tar'))

def train(train_loader, det2_dataset, model, model_ema, lr_scheduler, optimizer, logger, epoch, args, cfg):

    model.train()
    pixel_mean = cfg.MODEL.PIXEL_MEAN
    pixel_std = cfg.MODEL.PIXEL_STD
    # pixel_mean = [123.675, 116.28, 103.53]
    # pixel_std = [58.395, 57.12, 57.375]
    pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1).cuda()
    pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1).cuda()

    device = torch.device('cuda:{}'.format(args.local_rank))
    data_time = AverageMeter()
    batch_time = AverageMeter()

    loss_meter_dict = OrderedDict()
    loss_meter_dict['total'] = AverageMeter()
    loss_meter_dict['semantic'] = AverageMeter()
    loss_meter_dict['center'] = AverageMeter()
    loss_meter_dict['offset'] = AverageMeter()

    # for i, sample in enumerate(train_loader):
    for i in range(len(train_loader)):
        start_time = time.time()
        cur_iter = epoch * len(train_loader) + i
        lr_scheduler(optimizer, cur_iter)
        lr = lr_scheduler.get_lr(optimizer)

        # data = to_cuda(sample, device)
        # data_time.update(time.time() - start_time)

        det2_data = next(det2_dataset)
        det2_inputs = [x["image"].cuda(non_blocking=True) for x in det2_data]
        det2_inputs = [(x - pixel_mean) / pixel_std for x in det2_inputs]
        det2_inputs = ImageList.from_tensors(det2_inputs, 0).tensor
        data = {}
        data['image'] = det2_inputs

        det2_targets = [x["sem_seg"].cuda(non_blocking=True) for x in det2_data]
        det2_targets = ImageList.from_tensors(det2_targets, 0, args.ignore).tensor
        data['semantic'] = det2_targets
        
        det2_targets = [x["sem_seg_weights"].cuda(non_blocking=True) for x in det2_data]
        det2_targets = ImageList.from_tensors(det2_targets, 0, args.ignore).tensor
        data['semantic_weights'] = det2_targets
        
        other_keys = ['center', 'center_weights', 'offset', 'offset_weights'] 
        for k in other_keys:
            tmp_data = [x[k].cuda(non_blocking=True) for x in det2_data]
            tmp_data = ImageList.from_tensors(tmp_data, 0, args.ignore).tensor.squeeze(1)
            data[k] = tmp_data
         
        data['center'][data['center']==255] = 0.
        torch.cuda.synchronize()
        image = data.pop('image')

        loss_dict = model(image, data)
        loss = loss_dict['total']
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        torch.cuda.synchronize()
        for key in loss_dict.keys():
            reduce_loss = reduce_tensor(loss_dict[key].detach().data, args.world_size)
            loss_meter_dict[key].update(reduce_loss.cpu().item(), image.size(0))
        batch_time.update(time.time() - start_time)
        
        if args.local_rank == 0 and i % 20 == 0:
            msg = '[{0}/{1}][{2}/{3}] LR: {4:.7f}\t' \
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                    'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'.format(
                    i + 1, len(train_loader), epoch+1, args.epochs, lr, batch_time=batch_time, data_time=data_time)
            msg += get_loss_info_str(loss_meter_dict)
            logging.info(msg)

        torch.cuda.synchronize()

        if args.local_rank == 0:
            logger.add_scalar("LR", lr, cur_iter)
            for key in loss_meter_dict.keys():
                logger.add_scalar("Loss/{}".format(key), loss_meter_dict[key].val, cur_iter)

        if model_ema is not None:
            model_ema.update(model)

def validation(val_loader, model, criterion, n_classes, args, cal_miou=True):
    device = torch.device('cuda:{}'.format(args.local_rank))
    model.eval()
    test_loss = 0.0

    hist_size = (n_classes, n_classes)
    hist = torch.zeros(hist_size, dtype=torch.float32).cuda()

    for i, sample in enumerate(val_loader):
        sample = to_cuda(sample, device)
        image = sample['image']
        if args.dataset == "coco":
            target = sample['semantic']
            # target = sample['label'].long()
        else:
            target = sample['semantic']
        N, H, W = target.shape
        probs = torch.zeros((N, n_classes, H, W)).cuda()
        probs.requires_grad = False

        torch.cuda.synchronize()
        if args.local_rank==0:
            logging.info("Evaluation [{}/{}]".format(i+1, len(val_loader)))
        with torch.no_grad():
            output = model(image)
            prob = F.softmax(output, 1)
            probs += prob
            loss = criterion(output, target).detach().data
            dist.all_reduce(loss, dist.ReduceOp.SUM)
            test_loss += loss

            if args.eval_flip:
                output = model(torch.flip(image, dims=(3,)))
                output = torch.flip(output, dims=(3,))
                prob = F.softmax(output, 1)
                probs += prob
                loss = criterion(output, target).detach().data
                dist.all_reduce(loss, dist.ReduceOp.SUM)
                test_loss += loss

        if cal_miou:
            # probs = probs.data.numpy()
            preds = torch.argmax(probs, dim=1)
            hist_once = compute_hist(preds, target, n_classes, args.ignore)
            hist = hist + hist_once
        
        torch.cuda.synchronize()


    if args.eval_flip:
        avg_loss = test_loss / 2*len(val_loader)
    else:
        avg_loss = test_loss / len(val_loader)

    if cal_miou:
        # hist = torch.tensor(hist).cuda()
        dist.all_reduce(hist, dist.ReduceOp.SUM)
        hist = hist.cpu().numpy().astype(np.float32)
        IOUs = np.diag(hist) / (np.sum(hist, axis=0) + np.sum(hist, axis=1) - np.diag(hist))
        mIOU = np.mean(IOUs)
    else:
        mIOU = -avg_loss

    return mIOU*100, avg_loss


def validation_np(val_loader, model, criterion, n_classes, args, cal_miou=True):
    device = torch.device('cuda:{}'.format(args.local_rank))
    model.eval()
    test_loss = 0.0
    hist_size = (n_classes, n_classes)
    hist = np.zeros(hist_size, dtype=np.float32)

    for i, sample in enumerate(val_loader):
        sample = to_cuda(sample, device)
        image = sample['image']
        target = sample['semantic']
        N, H, W = target.shape
        probs = torch.zeros((N, n_classes, H, W))
        probs.requires_grad = False

        torch.cuda.synchronize()
        if args.local_rank==0:
            logging.info("Evaluation [{}/{}]".format(i+1, len(val_loader)))

        with torch.no_grad():
            output = model(image)
            prob = F.softmax(output, 1)
            probs += prob.cpu()
            loss = criterion(output, target).detach().data
            dist.all_reduce(loss, dist.ReduceOp.SUM)
            test_loss += loss

            if args.eval_flip:
                output = model(torch.flip(image, dims=(3,)))
                output = torch.flip(output, dims=(3,))
                prob = F.softmax(output, 1)
                probs += prob.cpu()
                loss = criterion(output, target).detach().data
                dist.all_reduce(loss, dist.ReduceOp.SUM)
                test_loss += loss

        if cal_miou:
            probs = probs.data.numpy()
            preds = np.argmax(probs, axis=1)
            hist_once = compute_hist_np(preds.flatten(), target.data.cpu().numpy().flatten(), n_classes, args.ignore)
            hist = hist + hist_once
        
        torch.cuda.synchronize()

    if args.eval_flip:
        avg_loss = test_loss / 2*len(val_loader)
    else:
        avg_loss = test_loss / len(val_loader)

    if cal_miou:
        hist = torch.tensor(hist).cuda()
        dist.all_reduce(hist, dist.ReduceOp.SUM)
        hist = hist.cpu().numpy().astype(np.float32)
        IOUs = np.diag(hist) / (np.sum(hist, axis=0) + np.sum(hist, axis=1) - np.diag(hist))
        mIOU = np.mean(IOUs)
    else:
        mIOU = -avg_loss

    return mIOU*100, avg_loss


if __name__ == '__main__':
    main() 
