from __future__ import division
import os
import os.path as osp
import sys
import time
import glob
import logging
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils
from tensorboardX import SummaryWriter

import numpy as np
# Force matplotlib to not use any Xwindows backend.
from thop import profile

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.automl import GeneticSearch
from ray.tune.automl import ContinuousSpace, DiscreteSpace, SearchSpace

"""Path Config"""
root_dir = "/home/hongyuan/test/FasterSeg"
this_dir = "/home/hongyuan/test/FasterSeg/search"
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(root_dir, 'tools'))
add_path(this_dir)
print("this_dir: ", this_dir)
print("root_dir: ", root_dir)

from config_search import myconfig
from seg_opr.loss_opr import ProbOhemCrossEntropy2d
from dataloader import get_train_loader
from datasets import Cityscapes
from eval import SegEvaluator
from genotypes import PRIMITIVES
from utils.darts_utils import load
from model_search import Network_Multi_Path as Network
from model_seg import Network_Multi_Path_Infer_SPOS

model_config = myconfig

# create model
min_kept = int(model_config.batch_size * model_config.image_height * model_config.image_width // (16 * model_config.gt_down_sampling ** 2))
ohem_criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7, min_kept=min_kept, use_weight=False)
supernet = Network(model_config.num_classes, model_config.layers, ohem_criterion, Fch=model_config.Fch, width_mult_list=model_config.width_mult_list, prun_modes=model_config.prun_modes, stem_head_width=model_config.stem_head_width[0])
load(supernet, osp.join(this_dir, "weights.pt"))
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

evaluator = SegEvaluator(Cityscapes(data_setting, 'val', None), model_config.num_classes, model_config.image_mean,
                            model_config.image_std, supernet, model_config.eval_scale_array, model_config.eval_flip, 0, config=model_config,
                            verbose=False, save_path=None, show_image=False)


def infer(model, evaluator, Network_Multi_Path_Infer_SPOS):
    model.eval()
    # for two branchs, three branch is 4
    evaluator.out_idx = 2 * model._branch - 2
    latency = get_lantency(model, Network_Multi_Path_Infer_SPOS)
    _, mIoU = evaluator.run_online()
    # _, mIoU = evaluator.run_online_multiprocess()
    return mIoU, latency

def get_lantency(model, Network_Multi_Path_Infer_SPOS):
    input_size = (1, 3, 1024, 2048)
    net = Network_Multi_Path_Infer_SPOS(
        model.ops_compact, model.paths_compact, model.downs_compact, model.widths_compact, model.lasts,
        num_classes=model._num_classes, layers=model._layers, Fch=model._Fch, width_mult_list=model._width_mult_list, stem_head_width=model._stem_head_width)

    net = net.cuda()
    net.eval()
    # ms
    latency, _ = net.forward_latency(input_size[1:])
    return latency

def fitness(config):
    bad_path = False
    path1_downs_1 = config['path1_downs_1']
    path1_downs_2 = config['path1_downs_2']
    path1_cands = [config['path1_cand_{}'.format(i)] for i in range(model_config.layers)]
    path1_ratio_indexes = [config['path1_ratio_{}'.format(i)] for i in range(model_config.layers)]

    path2_last = config['path2_last']
    path2_downs = config['path2_downs']
    path2_cands = [config['path2_cand_{}'.format(i)] for i in range(model_config.layers)]
    path2_ratio_indexes = [config['path2_ratio_{}'.format(i)] for i in range(model_config.layers)]

    if path1_downs_1 == path1_downs_2:
        path1_downs_index = [path1_downs_1, max(path1_downs_1-1, 0)]
    else:
        path1_downs_index = [path1_downs_1, path1_downs_2]
    if path2_last == 0:
        path2_downs_index = []
    else:
        path2_downs_index = [path2_downs]

    valid_path1 = evaluator.network.sample_from_seed(path1_downs_index, path1_cands, path1_ratio_indexes, branch_idx=0)
    valid_path2 = evaluator.network.sample_from_seed(path2_downs_index, path2_cands, path2_ratio_indexes, branch_idx=1)

    if (not valid_path1) or (not valid_path2):
        bad_path = True

    if bad_path:
        mIoU = -99999
    else:
        mIoU, latency = infer(supernet, evaluator, Network_Multi_Path_Infer_SPOS)
        # mIoU, latency = 0.1, 0.2
    print("model_arch: {}".format(evaluator.network.paths_compact))
    print("ops: {}".format(evaluator.network.ops_compact))
    print("widths: {}".format(evaluator.network.widths_compact))
    if bad_path:
        tune.report(mIoU=-9999)
    else:
        tune.report(mIoU=mIoU)


if __name__ == "__main__":

    # ray.init(num_gpus=args.gpu_nums, num_cpus=args.gpu_nums * 4, object_store_memory=50000000000)
    ray.init()

    path1_downs = {DiscreteSpace("path1_downs_1", [i for i in range(model_config.layers - 1)]),
                   DiscreteSpace("path1_downs_2", [i for i in range(model_config.layers - 1)])}
    path1_cands = {DiscreteSpace("path1_cand_{}".format(j), [i for i in range(len(PRIMITIVES))]) for j in range(model_config.layers-1)}
    path1_ratios = {DiscreteSpace("path1_ratio_{}".format(j), [i for i in range(len(model_config.width_mult_list))]) for j in range(model_config.layers)}

    path2_last = {DiscreteSpace("path2_last", [0, 1])}
    path2_downs = {DiscreteSpace("path2_downs", [i for i in range(model_config.layers - 1)])}
    path2_cands = {DiscreteSpace("path2_cand_{}".format(j), [i for i in range(len(PRIMITIVES))]) for j in range(model_config.layers-1)}
    path2_ratios = {DiscreteSpace("path2_ratio_{}".format(j), [i for i in range(len(model_config.width_mult_list))]) for j in range(model_config.layers)}

    paths_cands_last = {DiscreteSpace("path1_cand_{}".format(model_config.layers-1), [i for i in range(1, len(PRIMITIVES))]),
                        DiscreteSpace("path2_cand_{}".format(model_config.layers-1), [i for i in range(1, len(PRIMITIVES))])}
    # if channel constrain, we could sample breakpoints of different channel

    # merge sets
    all_sets = path1_downs | path1_cands | path1_ratios | path2_last | path2_downs | path2_cands | path2_ratios | paths_cands_last
    space = SearchSpace(all_sets)

    resources_per_trial = {
                "cpu": 1,  # single task cpu num
                "gpu": 0.5,  # single task gpu num
    }

    stop = {
        "mIoU": 0.5,  # if mIoU >= 0.5, this procedures will stop
        # "timesteps_total": 1000, # iteration times
        "training_iteration": 100,
    }
    config = {"stop": stop}


    algo = GeneticSearch(
        space,
        reward_attr="mIoU",
        max_generation=20,
        population_size=50)

    scheduler = AsyncHyperBandScheduler(metric="mIoU", mode="max")
    tune.run(fitness, name="GA", search_alg=algo, scheduler=scheduler, resources_per_trial=resources_per_trial, **config)


