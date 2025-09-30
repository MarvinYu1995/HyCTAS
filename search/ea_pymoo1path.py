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
from thop import profile

from config_search import config
from seg_opr.loss_opr import ProbOhemCrossEntropy2d
from dataloader import get_train_loader_norm
from datasets import Cityscapes
from eval import SegEvaluator
from genotypes import PRIMITIVES
from model_search import Network_Multi_Path as Network
from model_seg import Network_Multi_Path_Infer_SPOS
# from ptflops import get_model_complexity_info

from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from pymoo.model.mutation import Mutation
from pymoo.model.sampling import Sampling
from pymoo.model.crossover import Crossover
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.population import Population
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination

import _init_paths
from utils.pyt_utils import compute_hist, compute_hist_np, load_pretrain
from utils.init_func import NpEncoder

parser = argparse.ArgumentParser(description='parameters for sampling')
parser.add_argument('--resume', default='../../ep120_sk50_jp40_bs4/weights.pth.tar', type=str, help='resumed model')
parser.add_argument('--sk_ratio', type=float, default=0.5)
parser.add_argument('--jp_ratio', type=float, default=0.4)
# parser.add_argument('--resume', default='./weights.pth.tar', type=str, help='resumed model')
parser.add_argument('--save_dir', default='./3paths_latency_ep120_sk50_jp40_bs4', type=str, help='saved dict')
parser.add_argument('--target', type=int, default=0, help="0 latency 1 flops 2 params")
parser.add_argument('--gpu_num', default=4, type=int, help='gpu numbers')
parser.add_argument('--path_num', default=1, type=int, help='3 path')
# for supernet
parser.add_argument('--down_sampling', type=int, default=2, help='first down_sampling then crop')
parser.add_argument('--gt_down_sampling', type=int, default=8, help='model default output size without final upsamplingp')
parser.add_argument('--sepbn', type=bool, default=True)
args = parser.parse_args()

def decode_arch(downs, cands, ratios, layers=20):
    ops = []
    downs_compact = []
    path_compact = []
    widths_compact = []

    path = [sum(downs[:idx]) for idx in range(layers)]
    for i in range(layers):
        if cands[i] == 0:
            if i == layers - 1:
                widths_compact.pop()
            continue
        else:
            ops.append(cands[i])
            downs_compact.append(downs[i])
            path_compact.append(path[i])
            if i < (layers - 1):
                widths_compact.append(ratios[i])
    assert len(path_compact) == len(widths_compact) + 1 or len(path_compact) + len(widths_compact) == 0, "path %d, widt0 %d"%(len(path_compact), len(widths_compact))
    return ops, path_compact, downs_compact, widths_compact

def is_valid(seeds, width_mult_list, layers=20, path_var=33, path_index=[0, 1, 0]):
    lasts = [] # [0, 1]
    _downs = [] # 0 keep, 1 down 
    _cands = []
    _ratios = []
    # decoded archs
    _ops_compact = []
    _paths_compact = []
    _downs_compact = []
    _widths_compact = []

    path_index = np.array(path_index)
    branch_num = sum(path_index)
    if branch_num == 0:
        return False
    elif branch_num == 1:
        locate = np.where(path_index>0)[0][0]
        seeds = seeds[locate*path_var:(locate+1)*path_var]
    elif branch_num == 2:
        locate1 = np.where(path_index>0)[0][0]
        locate2 = np.where(path_index>0)[0][1]
        seeds = np.concatenate((seeds[locate1*path_var:(locate1+1)*path_var], seeds[locate2*path_var:(locate2+1)*path_var]))
    else:
        pass

    for i in range(branch_num):
        downs = np.zeros(layers, dtype=int)
        downs_index = seeds[i*path_var : i*path_var+2]
        cands = seeds[i*path_var+2 : i*path_var+2+layers]
        widths = seeds[i*path_var+2+layers : (i+1)*path_var]
        ratios = [width_mult_list[width_index] for width_index in widths]

        if downs_index[0] == downs_index[1] and downs_index[0] >= 0:
            return False
            # downs_index = np.random.choice(range(0, self._layers-1), 2, replace=False)

        for down_index in downs_index:
            if down_index >= 0:
                downs[down_index] = 1
            if cands[down_index] == 0:
                # cands[down_index] = 1
                return False
        lasts.append(sum(downs))
        ops_compact, path_compact, downs_compact, widths_compact = decode_arch(downs, cands, ratios)

        if downs_compact[-1] == 1:
            return False

        _downs.append(downs)
        _cands.append(cands)
        _ratios.append(ratios)
        _ops_compact.append(ops_compact)
        _paths_compact.append(path_compact)
        _downs_compact.append(downs_compact)
        _widths_compact.append(widths_compact)
    
    if branch_num == 2:
        if lasts[0] == lasts[1]:
            return False
    elif branch_num == 3:
        if set(lasts) == set([0,1,2]):
            pass
        else:
            return False
    else:
        pass
    return True

def random_sample_1path(width_mult_list, sk_ratio=0.5, last=2, layers=20, channel_constrain=False):
    downs = [-1, -1]
    cands = np.zeros(layers, dtype=int)
    ratios = np.zeros(layers-1, dtype=int)
   
    if last>0:
        down_index = np.random.choice(range(0, layers-1), last, replace=False)
        if len(down_index) == 0:
            pass
        elif len(down_index) == 1:
            downs[0] = down_index[0]
        else:
            downs[0] = down_index[0]
            downs[1] = down_index[1]

    for i in range(layers):
        if (i in downs) or (i == layers-1):
            cands[i] = np.random.randint(1, len(PRIMITIVES))
        else:
            p = np.random.rand()
            if p > sk_ratio:
                # cands[i] = np.random.randint(0, len(PRIMITIVES))
                cands[i] = np.random.randint(1, len(PRIMITIVES))
            else:
                cands[i] = 0
    
    index_l = 0
    index_h = len(width_mult_list)
    for j in range(layers - 1):
        width_index = np.random.randint(index_l, index_h)
        if channel_constrain:
            index_l = max(index_l, width_index)
        # ratios[j] = width_mult_list[width_index]
        ratios[j] = width_index

    return downs, cands, ratios

def sk_change(rand_int=0, sk_ration=0.5, sk_jump=0.3):
    if rand_int % 3 == 0:
        sk_ratio = sk_ration
    elif rand_int % 3 == 1:
        sk_ratio = max(0., sk_ration - sk_jump)
    else:
        sk_ratio = min(1.0, sk_ration + sk_jump)
    return sk_ratio

def init_population(sk_ratio_balance=0.5, sk_jump=0.3, population_num=120, layers=20, path_num=3, down_num=2, down_choice=15, cand_num=16, cand_choice=5, width_num=15, width_choice=5, path_var=33):
    width_mult_list = [4./12, 6./12, 8./12, 10./12, 1.]
    init_population = []
    cand_num = 0
    max_path_num = 3
    sk_ratio = None

    while cand_num <= population_num:
        seeds = []
        lasts = np.zeros(3, dtype=np.int)
        if path_num == 1:
            lasts_index = [np.random.randint(0, max_path_num)]
        elif path_num == 2:
            lasts_index = np.random.choice([0, 1, 2], 2, replace=False).tolist()
        elif path_num ==3:
            lasts_index = [0, 1, 2]
        else:
            raise Exception("Not support!")

        lasts[lasts_index] = 1 
        seeds = seeds + list(lasts)

        for branch_idx in range(path_num):
            if sk_ratio is None:
                sk_ratio = sk_change(np.random.randint(max_path_num), sk_ratio_balance, sk_jump)

            downs, cands, ratios = random_sample_1path(width_mult_list, sk_ratio, lasts_index[branch_idx], layers)
           
            seeds = seeds + list(downs)
            seeds = seeds + list(cands.squeeze())
            seeds = seeds + list(ratios.squeeze())
       
        # print(sk_ratio) 
        is_valid_flag = is_valid(seeds[max_path_num:], width_mult_list, layers, path_var, path_index=[1])
        # if not is_valid_flag:
        #     print(is_valid_flag)
        if is_valid_flag:
            sk_ratio = None
            cand_num += 1
            init_population.append(seeds)
        else:
            pass
        # print(sk_ratio) 
    return np.array(init_population)

class SposSeg(Problem):
    # first define the NAS problem (inherit from pymop)
    def __init__(self, supernet, evaluator, train_loader, save_dir, target, path_num=2, down_num=2, down_choice=15, cand_num=16, cand_choice=5,
                 width_num=15, width_choice=5, path_var=33, n_var=33*3+3, n_obj=2, n_constr=6):
        
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, type_var=np.int)

        self.path_var = path_var
        self.down_num = down_num
        self.cand_num = cand_num
        self.width_num = width_num
        self.path_num = path_num
        self.target = target

        self.max_path_num = 3
        self.fps_low = 20.
        self.fps_high = 200.

        self._supernet = supernet
        self._train_loader = train_loader
        self._evaluator = evaluator
        self._save_dir = save_dir
        self._n_evaluated = 0  # keep track of how many architectures are sampled

        self.xl = np.zeros(self.n_var, dtype=np.int)
        self.xu = np.ones(self.n_var, dtype=np.int)

        # the first 3 define how many paths
        for i in range(self.path_num):
            index_l = i*path_var + self.max_path_num
            index_h = i*path_var + down_num + self.max_path_num
            self.xl[index_l:index_h] = -1 # -1 means no path
            self.xu[index_l:index_h] = down_choice - 1
            index_l = index_h
            index_h = index_h + cand_num
            self.xu[index_l:index_h] = cand_choice - 1
            self.xl[index_h-1] = 1
            index_l = index_h
            index_h = index_h + width_num
            self.xu[index_l:index_h] = width_choice - 1

    def _evaluate(self, x, out, *args, **kwargs):

        objs = np.full((x.shape[0], self.n_obj), np.nan)

        for i in range(x.shape[0]):
            arch_id = self._n_evaluated + 1
            print('\n')
            logging.info('Network id = {}'.format(arch_id))
           
            is_valid = self._supernet.ea_decode(x[i, self.max_path_num:], self.path_var, [1])
            
            if is_valid:
                latency, flops, params = self.get_lantency(self._supernet)
                if self.target == 0:
                    if 1000./latency > self.fps_high or 1000./latency < self.fps_low:
                        mIoU, latency, flops, params = 0.0, 99999, 99999, 99999
                    else:
                        self.set_bn(self._supernet, self._train_loader)
                        mIoU = self.infer(self._supernet, self._evaluator)
                else:
                    self.set_bn(self._supernet, self._train_loader)
                    mIoU = self.infer(self._supernet, self._evaluator)
            else:
                mIoU, latency, flops, params = 0.0, 99999, 99999, 99999

            # all objectives assume to be MINIMIZED !!!!!
            objs[i, 0] = 100 * (1 - mIoU)
            if self.target == 0:
                objs[i, 1] = latency * 0.1
            elif self.target == 1:
                objs[i, 1] = flops
            elif self.target == 2:
                objs[i, 1] = params * 10
            else:
                raise NotImplementedError

            sample_dict = {}
            sample_dict["ops"] = self._supernet.ops_compact
            sample_dict["paths"] = self._supernet.paths_compact
            sample_dict["downs"] = self._supernet.downs_compact
            sample_dict["widths"] = self._supernet.widths_compact
            sample_dict["lasts"] = self._supernet.lasts
            sample_dict["mIoU"] = mIoU
            sample_dict["latency"] = latency
            sample_dict["flops"] = flops
            sample_dict["params"] = params

            with open(osp.join(self._save_dir, "{}.json".format(arch_id)), 'w') as f:
                json.dump(sample_dict, f, cls=NpEncoder)

            self._n_evaluated += 1

        out["F"] = objs
        # if your NAS problem has constraints, use the following line to set constraints
        # out["G"] = np.column_stack([g1, g2, g3, g4, g5, g6]) in case 6 constraints
        # g1 the downs location should be different
        g1 = (x[:, 0] == x[:, 1]) - 0.5
        # if downs, skip is forbidden
        downs11 = x[:, 0]
        downs12 = x[:, 1]
        cand1_index_l = self.down_num
        cand1_index_h = self.down_num + self.cand_num
        g2 = (x[:, cand1_index_l:cand1_index_h][np.arange(x.shape[0]), downs11] == 0) - 1
        g3 = (x[:, cand1_index_l:cand1_index_h][np.arange(x.shape[0]), downs12] == 0) - 1

        out["G"] = np.column_stack([g1, g2, g3])

    def infer(self, model, evaluator):
        model.eval()
        # for two branchs, three branch is 4
        evaluator.out_idx = 0
        mIoU = 0.
        _, mIoU = evaluator.run_online_multiprocess()
        # _, mIoU = evaluator.run_online()
        return mIoU

    def get_lantency(self, model):
        input_size = (1, 3, 1024, 2048)
        net = Network_Multi_Path_Infer_SPOS(
            model.ops_compact, model.paths_compact, model.downs_compact, model.widths_compact, model.lasts,
            num_classes=model._num_classes, layers=model._layers, Fch=model._Fch, width_mult_list=model._width_mult_list, stem_head_width=model._stem_head_width)

        net = net.to('cpu')
        # net = net.cuda()
        net.eval()
        try:
            flops, params = profile(net, inputs=(torch.randn(1, 3, 512, 1024),), verbose=False)
            flops = 4 * flops / 1e9
            params = params / 1e6

            net = net.cuda()
            net.eval()
            # ms
            latency, _ = net.forward_latency(input_size[1:])
            # flops, params = get_model_complexity_info(net, input_size[1:], as_strings=True,
            #                                     print_per_layer_stat=False, verbose=True)
            del net
        except:
            latency, flops, params = 99999, 99999, 99999
        return latency, flops, params

    def set_bn(self, model, train_loader):
        model.train()
        cal_iterrations = 200
        print("Set BN...")
        tbar = tqdm(range(cal_iterrations), ncols=80)
        train_loader = iter(train_loader)

        with torch.no_grad():
            for i in tbar:
                sample = next(train_loader)
                if i >= cal_iterrations:
                    break
                # imgs = sample['image'].cuda()
                # target = sample['semantic'].cuda()
                imgs = sample['data'].cuda()
                target = sample['label'].cuda()
                model.setbn_loss(imgs, target)


if __name__ == "__main__":
    # torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:{}'.format(np.random.randint(10000, 49999)), world_size=1, rank=0)
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


    path_var = 2 + model_config.layers + model_config.layers - 1
    max_path_num = 3
    n_var = path_var * args.path_num + max_path_num

    my_init_population = init_population(args.sk_ratio, args.jp_ratio, 120, model_config.layers, args.path_num, 2, model_config.layers-1, model_config.layers, len(PRIMITIVES), model_config.layers-1, len(model_config.width_mult_list), path_var)

    my_problem = SposSeg(supernet, evaluator, train_loader, args.save_dir, args.target, args.path_num, 2, model_config.layers-1,
        model_config.layers, len(PRIMITIVES), model_config.layers-1, len(model_config.width_mult_list), path_var, n_var=n_var)

    algorithm = NSGA2(
        pop_size=40,
        # n_offsprings=10,
        # sampling=get_sampling("int_random"),
        sampling=my_init_population,
        crossover=get_crossover("int_sbx", prob=0.9),
        mutation=get_mutation("int_pm", eta=1.0),
        eliminate_duplicates=True
    )
    termination = get_termination("n_gen", 20)

    res = minimize(my_problem,
               algorithm,
               termination,
               seed=1,
               save_history=False,
               verbose=True)



