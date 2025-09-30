from __future__ import division
import json
import os
import sys
import time
import glob
import logging
import yaml
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils
from apex import amp
from tensorboardX import SummaryWriter

import numpy as np
import matplotlib
import os.path as osp
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from thop import profile


from config_search import config
from dataloader import get_train_loader, get_train_loader_norm
from datasets import Cityscapes

import dataloaders
from utils.init_func import init_weight
from utils.lr_scheduler import Iter_LR_Scheduler
from seg_opr.loss_opr import ProbOhemCrossEntropy2d
from eval import SegEvaluator

from utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat
from model_search import Network_Multi_Path as Network
from model_seg import Network_Multi_Path_Infer_SPOS
from utils.init_func import NpEncoder


# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument("--epochs", default=120, type=int)
parser.add_argument('--sk_ratio', type=float, default=0.5, help='base learning rate')
parser.add_argument('--sk_jump', type=float, default=0.4, help='base learning rate')
parser.add_argument('--bs', type=int, default=4, help='batch size')
parser.add_argument('--save', type=str, default='ep120_sk50_jp40', help='save')
parser.add_argument('--resume', type=str, default=None, help='resume')

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

def main():
    # torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:{}'.format(np.random.randint(10000, 49999)), world_size=1, rank=0)

    args, args_text = _parse_args()
    config.save = os.path.join("../OUTPUT/", args.save)
    config.nepochs = args.epochs
    config.batch_size = args.bs
    config.train_samples = os.path.join(config.save, "train_samples")
    os.makedirs(config.train_samples, exist_ok=True)
    create_exp_dir(config.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh'))
    logger = SummaryWriter(config.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info("args = %s", str(config))
    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # config network and criterion ################
    min_kept = int(config.batch_size * config.image_height * config.image_width // (16 * config.gt_down_sampling ** 2))
    ohem_criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7, min_kept=min_kept, use_weight=False)

    # Model #######################################
    model = Network(config.num_classes, config.layers, ohem_criterion, Fch=config.Fch, sepbn=config.sepbn, width_mult_list=config.width_mult_list, prun_modes=config.prun_modes, stem_head_width=config.stem_head_width[0], sk_ratio=args.sk_ratio, sk_jump=args.sk_jump)
    model.sample_arch()
    flops, params = profile(model, inputs=(torch.randn(1, 3, 1024, 2048),), verbose=False)
    logging.info("params = %fMB, FLOPs = %fGB", params / 1e6, flops / 1e9)
    model = model.cuda()
    
    init_weight(model, nn.init.kaiming_normal_, nn.BatchNorm2d, config.bn_eps, config.bn_momentum, mode='fan_in', nonlinearity='relu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer ###################################
    base_lr = config.lr
    parameters = []
    parameters += list(model.parameters())
    #parameters += list(model.cells.parameters())
    #parameters += list(model.refine32.parameters())
    #parameters += list(model.refine16.parameters())
    #parameters += list(model.head0.parameters())
    #parameters += list(model.head1.parameters())
    #parameters += list(model.head2.parameters())
    #parameters += list(model.head02.parameters())
    #parameters += list(model.head12.parameters())
    optimizer = torch.optim.SGD(
        parameters,
        lr=base_lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay)
    
    # lr policy ##############################
    # lr_policy = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.978)
    if config.nepochs <= 120:
        lr_policy = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.98)
    elif config.nepochs <= 300:
        lr_policy = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.992)
    elif config.nepochs <= 600:
        lr_policy = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.996)
    else:
        raise NotImplementedError

    # data loader ###########################
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'down_sampling': config.down_sampling}
    # train_loader_model, _ = get_train_loader(config, Cityscapes, portion=config.train_portion)
    train_loader_model  = get_train_loader_norm(config, Cityscapes, portion=config.train_portion)

    evaluator = SegEvaluator(Cityscapes(data_setting, 'val', None), config.num_classes, config.image_mean,
                             config.image_std, model, config.eval_scale_array, config.eval_flip, 0, config=config,
                             verbose=False, save_path=None, show_image=False)

    # tbar = tqdm(range(config.nepochs), ncols=80)
    valid_mIoU_history = []; FPSs_history = [];
    valid_names = ["8s", "16s", "32s"]
    arch_name = "spos"
    arch_id = 0

    ### amp ###
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    start_epoch = 0
    if os.path.exists(os.path.join(config.save, 'weights.pth.tar')):
        args.resume = os.path.join(config.save, 'weights.pth.tar')
    else:
        args.resume = None

    if args.resume:
        model_state_file = args.resume
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location=torch.device('cpu'))
            start_epoch = checkpoint['start_epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_policy.load_state_dict(checkpoint['lr_scheduler'])
            amp.load_state_dict(checkpoint['amp'])
            logging.info('Loaded checkpoint (starting from iter {})'.format(checkpoint['start_epoch']))

    for epoch in range(start_epoch, config.nepochs):
        logging.info(config.save)
        logging.info("lr: " + str(optimizer.param_groups[0]['lr']))
        # training
        logging.info("[Epoch %d/%d][train...]" % (epoch + 1, config.nepochs))
        arch_id = train(train_loader_model, model, ohem_criterion, optimizer, lr_policy, logger, epoch, arch_id)
        torch.cuda.empty_cache()
        lr_policy.step()

        # validation
#        tbar.set_description("[Epoch %d/%d][validation...]" % (epoch + 1, config.nepochs))
#        with torch.no_grad():
#            valid_mIoUs = []; FPSs = []
#            model.prun_mode = None
#            evaluator.network.sample_arch()
#
#            valid_mIoU, fps = infer(epoch, model, evaluator, logger)
#            # valid_mIoUs = infer(epoch, model, evaluator, logger, FPS=False)
#            valid_mIoUs.append(valid_mIoU)
#            FPSs.append(fps)
#
#            logger.add_scalar('mIoU/val_%s_%s'%(arch_name, "val"), valid_mIoU, epoch)
#            logging.info("Epoch %d: valid_mIoU_%s_%s %.3f"%(epoch, arch_name, "val", valid_mIoU))
#
#            logger.add_scalar('Objective/val'%objective_acc_lat(valid_mIoU, 1000./fps), epoch)
#            logging.info("Epoch %d: Objective %.3f"%(epoch, objective_acc_lat(valid_mIoU, 1000./fps)))
#
#            valid_mIoU_history.append(valid_mIoUs)
#            FPSs_history.append(FPSs)
#
        torch.save({
            'start_epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_policy.state_dict(),
            'amp': amp.state_dict(),
        }, os.path.join(config.save, 'weights.pth.tar'))

        # save(model, os.path.join(config.save, 'weights.pt'))

def train(train_loader_model, model, criterion, optimizer, lr_policy, logger, epoch, arch_id):
    model.train()

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format, ncols=80)
    dataloader_model = iter(train_loader_model)

    for step in pbar:
        optimizer.zero_grad()


        #for i in range(400):
        #    model.sample_arch()
        #    latency = get_latency(model, Network_Multi_Path_Infer_SPOS)
        #    if latency < 9:
        #        break
        try:
            minibatch = dataloader_model.next()
            imgs = minibatch['data']
            target = minibatch['label']
            imgs = imgs.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            for i in range(400):
                seed_p = np.random.rand(1)[0]
                if seed_p<0.2:
                    sample_dict = model.sample_arch(branch_num=1)
                elif seed_p<0.5:
                    sample_dict = model.sample_arch(branch_num=2)
                else:
                    sample_dict = model.sample_arch(branch_num=3)

                latency = get_latency(model, Network_Multi_Path_Infer_SPOS)
                if latency < 100:
                    break

            loss = model._loss(imgs, target)
            logging.info("Loss: {}".format(loss))
            logger.add_scalar('loss/train', loss, epoch*len(pbar)+step)

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            # loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            with open(osp.join(config.train_samples, "{}.json".format(arch_id)), 'w') as f:
                json.dump(sample_dict, f, cls=NpEncoder)
                arch_id = arch_id + 1

            pbar.set_description("[Step %d/%d]" % (step + 1, len(train_loader_model)))
        except Exception as e:
            print(e)
            seed_p = 0.1
            torch.cuda.empty_cache()
            time.sleep(3)
            torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return arch_id
    # del loss
    # if update_arch: del loss_arch


def infer(epoch, model, evaluator, logger, FPS=True):
    model.eval()
    # evaluator.network.sample_arch()
    evaluator.out_idx = 0
    # _, mIoU = evaluator.run_online()
    _, mIoU = evaluator.run_online_multiprocess()
    if FPS:
        fps = arch_logging(model, config, logger, epoch)
        return mIoU, fps
    else:
        return mIoU

def arch_logging(model, args, logger, epoch):
    input_size = (1, 3, 1024, 2048)
    net = Network_Multi_Path_Infer_SPOS(
        model.ops_compact, model.paths_compact, model.downs_compact, model.widths_compact, model.lasts,
        num_classes=model._num_classes, layers=model._layers, Fch=model._Fch, width_mult_list=model._width_mult_list, stem_head_width=model._stem_head_width)

    for i in range(net._branch):
        plot_op(net.ops[i], net.paths[i], F_base=args.Fch).savefig("table.png", bbox_inches="tight")
        logger.add_image("arch/ops{}_arch".format(i), np.swapaxes(np.swapaxes(plt.imread("table.png"), 0, 2), 1, 2), epoch)

    latency = 100000

    net = net.cuda()
    net.eval()
    latency, _ = net.forward_latency(input_size[1:])
    logger.add_scalar("arch/fps_arch", 1000./latency, epoch)
    logger.add_figure("arch/path_width_arch_012", plot_path_width(net.lasts, net.paths, net.widths), epoch)


    return 1000./latency


def get_latency(model, model_infer):
    input_size = (1, 3, 1024, 2048)
    net = model_infer(
        model.ops_compact, model.paths_compact, model.downs_compact, model.widths_compact, model.lasts,
        num_classes=model._num_classes, layers=model._layers, Fch=model._Fch, width_mult_list=model._width_mult_list, stem_head_width=model._stem_head_width)

    net = net.cuda()
    net.eval()
    # ms
    latency, _ = net.forward_latency(input_size[1:])
    del net
    return latency

if __name__ == '__main__':
    main() 
