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

from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from pymoo.model.mutation import Mutation
from pymoo.model.sampling import Sampling
from pymoo.model.crossover import Crossover
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination

import _init_paths
from utils.pyt_utils import compute_hist, compute_hist_np, load_pretrain
from utils.init_func import NpEncoder

parser = argparse.ArgumentParser(description='parameters for sampling')
parser.add_argument('--resume', default='weights.pt', type=str, help='resumed model')
parser.add_argument('--save_dir', default='./321paths', type=str, help='saved dict')
parser.add_argument('--gpu_nums', default=4, type=int, help='gpu numbers')
# for supernet
parser.add_argument('--down_sampling', type=int, default=2, help='first down_sampling then crop')
parser.add_argument('--gt_down_sampling', type=int, default=8, help='model default output size without final upsamplingp')
parser.add_argument('--sepbn', type=bool, default=True)
args = parser.parse_args()

class SposSeg(Problem):
    # first define the NAS problem (inherit from pymop)
    def __init__(self, supernet, evaluator, train_loader, save_dir, max_path_num=3, down_num=2, down_choice=15, cand_num=16, cand_choice=5,
                 width_num=15, width_choice=5, path_var=33, n_var=33*3+3, n_obj=2, n_constr=5):
        
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, type_var=np.int)

        self.path_var = path_var
        self.down_num = down_num
        self.cand_num = cand_num
        self.width_num = width_num
        self.max_path_num = max_path_num

        self._supernet = supernet
        self._train_loader = train_loader
        self._evaluator = evaluator
        self._save_dir = save_dir
        self._n_evaluated = 0  # keep track of how many architectures are sampled

        self.xl = np.zeros(self.n_var, dtype=np.int)
        self.xu = np.ones(self.n_var, dtype=np.int)

        # the first 3 define how many paths
        for i in range(max_path_num):
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

        # fix downs
        # 0 down
        # self.xl[:down_num] = 0
        self.xu[self.max_path_num:down_num+self.max_path_num] = -1
        # 1 down
        self.xl[path_var+down_num-2+self.max_path_num] = 0
        self.xu[path_var+down_num-1+self.max_path_num] = -1
        # 2 downs
        # self.xu[2*path_var:2*path_var+down_num] = -1
        self.xl[2*path_var+self.max_path_num:2*path_var+down_num+self.max_path_num] = 0

    def _evaluate(self, x, out, *args, **kwargs):

        objs = np.full((x.shape[0], self.n_obj), np.nan)

        for i in range(x.shape[0]):
            arch_id = self._n_evaluated + 1
            print('\n')
            logging.info('Network id = {}'.format(arch_id))
            
            is_valid = self._supernet.ea_decode(x[i, self.max_path_num:], self.path_var, x[i, :self.max_path_num])
            
            if is_valid:
                self.set_bn(self._train_loader, self._supernet)
                mIoU, latency = self.infer(self._supernet, self._evaluator)
            else:
                mIoU, latency = 0.0, 99999

            # all objectives assume to be MINIMIZED !!!!!
            objs[i, 0] = 100 * (1 - mIoU)
            objs[i, 1] = latency * 0.1

            sample_dict = {}
            sample_dict["ops"] = self._supernet.ops_compact
            sample_dict["paths"] = self._supernet.paths_compact
            sample_dict["downs"] = self._supernet.downs_compact
            sample_dict["widths"] = self._supernet.widths_compact
            sample_dict["lasts"] = self._supernet.lasts
            sample_dict["mIoU"] = mIoU
            sample_dict["latency"] = latency

            with open(osp.join(self._save_dir, "{}.json".format(arch_id)), 'w') as f:
                json.dump(sample_dict, f, cls=NpEncoder)

            self._n_evaluated += 1

        out["F"] = objs
        # if your NAS problem has constraints, use the following line to set constraints
        # out["G"] = np.column_stack([g1, g2, g3, g4, g5, g6]) in case 6 constraints
        # g1 the downs location should be different
        g1 = (x[:, self.max_path_num+2*self.path_var] == x[:, 1+self.max_path_num+2*self.path_var]) - 0.5
        # if downs, skip is forbidden
        downs11 = x[:, self.max_path_num+2*self.path_var]
        downs12 = x[:, 1+self.max_path_num+2*self.path_var]
        downs21 = x[:, self.path_var+self.max_path_num]
        downs22 = x[:, self.path_var+1+self.max_path_num]
        cand1_index_l = self.down_num + self.max_path_num + 2*self.path_var
        cand1_index_h = self.down_num + self.cand_num + self.max_path_num + 2*self.path_var
        cand2_index_l = self.path_var + self.down_num + self.max_path_num
        cand2_index_h = self.path_var + self.down_num + self.cand_num + self.max_path_num
        g2 = (x[:, cand1_index_l:cand1_index_h][np.arange(x.shape[0]), downs11] == 0) - 1
        g3 = (x[:, cand1_index_l:cand1_index_h][np.arange(x.shape[0]), downs12] == 0) - 1
        g4 = (x[:, cand2_index_l:cand2_index_h][np.arange(x.shape[0]), downs21] == 0) + (x[:, cand2_index_l:cand2_index_h][np.arange(x.shape[0]), downs22] == 0) - 1.5

        # g5, path num should > 0
        g5 = 0.5 - (np.sum(x[:, 0:self.max_path_num], axis=1))
        out["G"] = np.column_stack([g1, g2, g3, g4, g5])

    def infer(self, model, evaluator):
        model.eval()
        # for two branchs, three branch is 4
        evaluator.out_idx = 0
        _, mIoU = evaluator.run_online_multiprocess()
        latency = self.get_lantency(model)
        # _, mIoU = evaluator.run_online()
        return mIoU, latency

    def get_lantency(self, model):
        input_size = (1, 3, 1024, 2048)
        net = Network_Multi_Path_Infer_SPOS(
            model.ops_compact, model.paths_compact, model.downs_compact, model.widths_compact, model.lasts,
            num_classes=model._num_classes, layers=model._layers, Fch=model._Fch, width_mult_list=model._width_mult_list, stem_head_width=model._stem_head_width)

        net = net.cuda()
        net.eval()
        # ms
        latency, _ = net.forward_latency(input_size[1:])
        return latency

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
    torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:26555', world_size=1, rank=0)
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
    n_var = path_var * model_config.branch + max_path_num
    my_problem = SposSeg(supernet, evaluator, train_loader, args.save_dir, max_path_num, 2, model_config.layers-1,
        model_config.layers, len(PRIMITIVES), model_config.layers-1, len(model_config.width_mult_list), path_var, n_var=n_var)

    algorithm = NSGA2(
        pop_size=40,
        # n_offsprings=10,
        sampling=get_sampling("int_random"),
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


