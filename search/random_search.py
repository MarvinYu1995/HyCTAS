from __future__ import division
import os
import os.path as osp
import sys
import time
import glob
import json
import logging
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils
from tensorboardX import SummaryWriter

import numpy as np
# Force matplotlib to not use any Xwindows backend.
from config_search import config
from thop import profile
from dataloader import get_train_loader_norm

from seg_opr.loss_opr import ProbOhemCrossEntropy2d
from datasets import Cityscapes
from eval import SegEvaluator
from genotypes import PRIMITIVES
from utils.darts_utils import load
from model_search import Network_Multi_Path as Network
from model_seg import Network_Multi_Path_Infer_SPOS

parser = argparse.ArgumentParser(description='parameters for sampling')
parser.add_argument('--resume', default='./weights.pt', type=str, help='resumed model')
parser.add_argument('--save_dir', default='./random_samples', type=str, help='saved dict')
parser.add_argument('--gpu_nums', default=4, type=int, help='gpu numbers')
parser.add_argument('--sample_nums', default=200, type=int, help='ramdom sample numbers')
parser.add_argument('--set_bn', action='store_false', default=True,
                    help='Recalculate BN statistics')
# for supernet
parser.add_argument('--path_num', default=1, type=int, help='sample_path num')
parser.add_argument('--down_sampling', type=int, default=2, help='first down_sampling then crop')
parser.add_argument('--gt_down_sampling', type=int, default=8, help='model default output size without final upsamplingp')
parser.add_argument('--sepbn', type=bool, default=True)

# others
parser.add_argument("--json_file", default=None, type=str, help='model_arch')

args = parser.parse_args()

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys

    print('missing keys:{}'.format(missing_keys))
    print('unused checkpoint keys:{}'.format(unused_pretrained_keys))
    # print('used keys:{}'.format(used_pretrained_keys))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    '''
    Old style model is stored with all names of parameters share common prefix 'module.'
    '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_pretrain(model, pretrained_path):
    print('load pretrained model from {}'.format(pretrained_path))

    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def main():
    torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:26445', world_size=1, rank=0)
    args.save_dir = args.save_dir + str(args.path_num)
    os.makedirs(args.save_dir, exist_ok=True)
    model_config = config
    # create model
    min_kept = int(model_config.batch_size * model_config.image_height * model_config.image_width // (16 * model_config.gt_down_sampling ** 2))
    ohem_criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7, min_kept=min_kept, use_weight=False)

    if args.gt_down_sampling > 1:
        upsampling = True
    else:
        upsampling = False

    supernet = Network(model_config.num_classes, model_config.layers, ohem_criterion, Fch=model_config.Fch, sepbn=model_config.sepbn, upsampling=upsampling, width_mult_list=model_config.width_mult_list, prun_modes=model_config.prun_modes, stem_head_width=model_config.stem_head_width[0])

    supernet = load_pretrain(supernet, args.resume)
    supernet.sample_arch()
    flops, params = profile(supernet, inputs=(torch.randn(1, 3, 1024, 2048),), verbose=False)
    logging.info("params = %fMB, FLOPs = %fGB", params / 1e6, flops / 1e9)
    supernet = supernet.cuda()
    print('pretrained model has been loaded')

    # data loader ###########################

    data_setting = {'img_root': model_config.img_root_folder,
                    'gt_root': model_config.gt_root_folder,
                    'train_source': model_config.train_source,
                    'eval_source': model_config.eval_source,
                    'down_sampling': model_config.down_sampling}

    train_loader = get_train_loader_norm(model_config, Cityscapes, portion=config.train_portion)
    evaluator = SegEvaluator(Cityscapes(data_setting, 'val', None), model_config.num_classes, model_config.image_mean,
                                model_config.image_std, supernet, model_config.eval_scale_array, model_config.eval_flip, 0, config=model_config,
                                verbose=False, save_path=None, show_image=False)

    if args.json_file:
        with open(args.json_file, 'r') as f:
            model_dict = json.loads(f.read())
        
        supernet.branch_num = len(model_dict['lasts'])
        supernet.downs = model_dict['downs']
        supernet.cands = model_dict['ops']
        supernet.ratios = model_dict['widths']
        supernet.lasts = model_dict['lasts']

        supernet.ops_compact = model_dict['ops']
        supernet.paths_compact = model_dict['paths']
        supernet.downs_compact = model_dict['downs']
        supernet.widths_compact = model_dict['widths']

        set_bn(train_loader, supernet)

        mIoU, latency, flops, params = infer(supernet, evaluator, Network_Multi_Path_Infer_SPOS)
        print("Set BN Sample mIoU: {} FPS: {}".format(mIoU, 1000./latency))

    outputs = []
    for i in range(args.sample_nums):
        sample_dict = {}
        supernet.sample_arch(args.path_num)

        sample_dict["ops"] = supernet.ops_compact
        sample_dict["paths"] = supernet.paths_compact
        sample_dict["downs"] = supernet.downs_compact
        sample_dict["widths"] = supernet.widths_compact
        sample_dict["lasts"] = supernet.lasts


        # mIoU, latency, flops, params = infer(supernet, evaluator, Network_Multi_Path_Infer_SPOS)
        # print("Sample index: {} mIoU: {} FPS: {} Flops: {} Params: {}".format(i, mIoU, 1000./latency, flops, params))



        # if args.set_bn:
        set_bn(train_loader, supernet)

        mIoU, latency, flops, params = infer(supernet, evaluator, Network_Multi_Path_Infer_SPOS)
        print("Set BN Sample index: {} mIoU: {} FPS: {}".format(i, mIoU, 1000./latency))
        # sample_dict["mIoU_bn"] = mIoU
        sample_dict["mIoU"] = mIoU
        sample_dict["latency"] = latency
        sample_dict["flops"] = flops
        sample_dict["params"] = params

        outputs.append(sample_dict)
        with open(osp.join(args.save_dir, "{}.json".format(i)), 'w') as f:
            json.dump(sample_dict, f, cls=NpEncoder)

    with open("all_results.json", 'w') as f:
        json.dump(outputs, f, cls=NpEncoder)

def set_bn(train_loader_model, model):
    model.train()
    cal_iterrations = 200
    dataloader_model = iter(train_loader_model)
    with torch.no_grad():
        for _ in range(cal_iterrations):
            minibatch = dataloader_model.next()
            imgs = minibatch['data']
            target = minibatch['label']
            imgs = imgs.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            model.setbn_loss(imgs, target)

def infer(model, evaluator, model_infer):
    model.eval()
    # for two branchs, three branch is 4
    evaluator.out_idx = 0
    latency, flops, params = get_model_info(model, model_infer)
    # _, mIoU = evaluator.run_online()
    _, mIoU = evaluator.run_online_multiprocess()
    return mIoU, latency, flops, params

def get_model_info(model, model_infer):
    input_size = (1, 3, 1024, 2048)
    net = model_infer(
        model.ops_compact, model.paths_compact, model.downs_compact, model.widths_compact, model.lasts,
        num_classes=model._num_classes, layers=model._layers, Fch=model._Fch, width_mult_list=model._width_mult_list, stem_head_width=model._stem_head_width)

    # flops, params = profile(net, inputs=(torch.randn(1, 3, 1024, 2048),), verbose=False)
    # flops = flops / 1e9
    # params = params / 1e6
    flops = 0.
    params = 0.
    net = net.cuda()
    net.eval()
    # ms
    latency, _ = net.forward_latency(input_size[1:])
    return latency, flops, params


if __name__ == '__main__':
    main() 
