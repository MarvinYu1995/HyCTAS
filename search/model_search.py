import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
# from utils.darts_utils import drop_path, compute_speed, compute_speed_tensorrt
from pdb import set_trace as bp
from seg_oprs import FeatureFusion, Head
import numpy as np
from slimmable_ops import USConv2d, USBatchNorm2d
from layers import NaiveSyncBatchNorm

# BatchNorm2d = NaiveSyncBatchNorm
BatchNorm2d = nn.BatchNorm2d

class MixedOp(nn.Module):

    def __init__(self, C_in, C_out, stride=1, width_mult_list=[1.], sepbn=False):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self._width_mult_list = width_mult_list
        self.sepbn = sepbn
        for primitive in PRIMITIVES:
            op = OPS[primitive](C_in, C_out, stride, True, width_mult_list=width_mult_list, sepbn=self.sepbn)
            self._ops.append(op)

    def set_prun_ratio(self, cand_idx, ratio):
        self._ops[cand_idx].set_ratio(ratio)
        # for op in self._ops:
        #     op.set_ratio(ratio)

    # def forward(self, x, weights, ratios):
    def forward(self, x, cand_idx, ratios, branch=0):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        ratio0 = ratios[0]
        ratio1 = ratios[1]
        self.set_prun_ratio(cand_idx, (ratio0, ratio1))
        x = self._ops[cand_idx](x, branch)
        return x

    # def forward_latency(self, size, weights, ratios):
    def forward_latency(self, size, cand_idx, ratios):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        ratio0 = self._width_mult_list[ratios[0]]
        ratio1 = self._width_mult_list[ratios[1]]
        self.set_prun_ratio((ratio0, ratio1))
        latency, size_out = self._ops[cand_idx].forward_latency(size)
        return latency, size_out

class Cell(nn.Module):
    def __init__(self, C_in, C_out=None, down=True, width_mult_list=[1.], sepbn=False):
        super(Cell, self).__init__()
        self._C_in = C_in
        if C_out is None: C_out = C_in
        self._C_out = C_out
        self._down = down
        self._width_mult_list = width_mult_list
        self.sepbn=sepbn

        self._op = MixedOp(C_in, C_out, width_mult_list=width_mult_list, sepbn=self.sepbn)

        if self._down:
            self.downsample = MixedOp(C_in, C_in*2, stride=2, width_mult_list=width_mult_list, sepbn=self.sepbn)
    
    # def forward(self, input, alphas, ratios):
    def forward(self, x, cand_idx, ratios, down=False, branch=0):
        # ratios: (in, out, down)
        if down:
            x = self.downsample(x, cand_idx, (ratios[0], ratios[1]), branch) if self._down else None
        else:
            x = self._op(x, cand_idx, (ratios[0], ratios[1]), branch)
        return x

    def forward_latency(self, size, cand_idx, ratios, down=False):
        # ratios: (in, out, down)
        if down:
            x = self.downsample.forward_latency(size, cand_idx, (ratios[0], ratios[1])) if self._down else None
        else:
            x = self._op.forward_latency(size, cand_idx, (ratios[0], ratios[1]))
        return x

class Network_Multi_Path(nn.Module):
    def __init__(self, num_classes=19, layers=16, criterion=nn.CrossEntropyLoss(ignore_index=-1), Fch=12, sepbn=False, upsampling=True, width_mult_list=[1.,], prun_modes=['arch_ratio',], stem_head_width=(1., 1.), sk_ratio=0.5, sk_jump=0.3):
        super(Network_Multi_Path, self).__init__()
        self._num_classes = num_classes
        assert layers >= 3
        self._layers = layers
        self._criterion = criterion
        self._Fch = Fch
        self._width_mult_list = width_mult_list
        self._prun_modes = prun_modes
        self.prun_mode = None # prun_mode is higher priority than _prun_modes
        self._stem_head_width = stem_head_width
        self._flops = 0
        self._params = 0
        self.sepbn=sepbn
        self.upsampling = upsampling
        self.sk_ratio = sk_ratio

        self.sk_ratio_balance = sk_ratio
        self.sk_jump = sk_jump

        stem_ratio = self._stem_head_width[0]
        self.stem = nn.Sequential(
                ConvNorm(3, self.num_filters(2, stem_ratio)*2, kernel_size=3, stride=2, padding=1, bias=False, groups=1, slimmable=False),
                BasicResidual2x(self.num_filters(2, stem_ratio)*2, self.num_filters(4, stem_ratio)*2, kernel_size=3, stride=2, groups=1, slimmable=False),
                BasicResidual2x(self.num_filters(4, stem_ratio)*2, self.num_filters(8, stem_ratio), kernel_size=3, stride=2, groups=1, slimmable=False)
        )

        self.cells = nn.ModuleList()
        for l in range(layers):
            cells = nn.ModuleList()
            if l == 0:
                # first node has only one input (prev cell's output)
                cells.append(Cell(self.num_filters(8), width_mult_list=width_mult_list, sepbn=self.sepbn))
            elif l == 1:
                cells.append(Cell(self.num_filters(8), width_mult_list=width_mult_list, sepbn=self.sepbn))
                cells.append(Cell(self.num_filters(16), width_mult_list=width_mult_list, sepbn=self.sepbn))
            elif l < layers - 1:
                cells.append(Cell(self.num_filters(8), width_mult_list=width_mult_list, sepbn=self.sepbn))
                cells.append(Cell(self.num_filters(16), width_mult_list=width_mult_list, sepbn=self.sepbn))
                cells.append(Cell(self.num_filters(32), down=False, width_mult_list=width_mult_list, sepbn=self.sepbn))
            else:
                cells.append(Cell(self.num_filters(8), down=False, width_mult_list=width_mult_list, sepbn=self.sepbn))
                cells.append(Cell(self.num_filters(16), down=False, width_mult_list=width_mult_list, sepbn=self.sepbn))
                cells.append(Cell(self.num_filters(32), down=False, width_mult_list=width_mult_list, sepbn=self.sepbn))
            self.cells.append(cells)

        # 0 min channel, 1 random channel, 2 max channel ref to sandwitch rule
        self.branch_num = 1 # 1-3

        # random parameters for spos training
        # self.last and self.downs specify the topological structure
        self.lasts = [] # [0, 1]
        self.downs = [] # 0 keep, 1 down 
        # self.cands defines which operation is sampled
        self.cands = []
        # self.ratios defines width
        self.ratios = []

        # decoded archs
        self.ops_compact = []
        self.paths_compact = []
        self.downs_compact = []
        self.widths_compact = []
       
        self.sample_arch()
        self.build_arm_ffm_head()
        
    def num_filters(self, scale, width=1.0):
        return int(np.round(scale * self._Fch * width))

    def build_arm_ffm_head(self):
        # list all possible heads
        self.heads32 = Head(self.num_filters(32, self._stem_head_width[1]), self._num_classes, True, norm_layer=BatchNorm2d)
        self.heads16_1 = Head(self.num_filters(16, self._stem_head_width[1]), self._num_classes, True, norm_layer=BatchNorm2d)
        self.heads16_2 = Head(self.num_filters(16, self._stem_head_width[1]) * 2, self._num_classes, True, norm_layer=BatchNorm2d)
        self.heads8_1 = Head(self.num_filters(8, self._stem_head_width[1]), self._num_classes, Fch=self._Fch, scale=4, branch=1, is_aux=False, norm_layer=BatchNorm2d)
        self.heads8_2 = Head(self.num_filters(8, self._stem_head_width[1]) * 2, self._num_classes, Fch=self._Fch, scale=4, branch=2, is_aux=False, norm_layer=BatchNorm2d)
        self.heads8_3 = Head(self.num_filters(8, self._stem_head_width[1]) * 3, self._num_classes, Fch=self._Fch, scale=4, branch=3, is_aux=False, norm_layer=BatchNorm2d)

        self.arms32 = nn.ModuleList([
            ConvNorm(self.num_filters(32, self._stem_head_width[1]), self.num_filters(16, self._stem_head_width[1]), 1, 1, 0, slimmable=False),
            ConvNorm(self.num_filters(16, self._stem_head_width[1]), self.num_filters(8, self._stem_head_width[1]), 1, 1, 0, slimmable=False),
        ])
        self.refines32 = nn.ModuleList([
            ConvNorm(self.num_filters(16, 1) * 2, self.num_filters(16, 1) * 2, 3, 1, 1, slimmable=True, width_mult_list=[self._stem_head_width[1]/2+x/2 for x in self._width_mult_list] + [self._stem_head_width[1]/2]),
            ConvNorm(self.num_filters(8, 1) * 2, self.num_filters(8, 1) * 2, 3, 1, 1, slimmable=True, width_mult_list=[self._stem_head_width[1]/2+x/2 for x in self._width_mult_list] + [self._stem_head_width[1]/2]),
        ])

        self.arms16 = ConvNorm(self.num_filters(16, self._stem_head_width[1]), self.num_filters(8, self._stem_head_width[1]), 1, 1, 0, slimmable=False)
        self.refines16 = ConvNorm(self.num_filters(8, 1) * 2, self.num_filters(8, 1) * 2, 3, 1, 1, slimmable=True, width_mult_list=[self._stem_head_width[1]/2+x/2 for x in self._width_mult_list] + [self._stem_head_width[1]/2])
        
        self.refines16_add = ConvNorm(self.num_filters(16, 1), self.num_filters(16, 1), 1, 1, 0, slimmable=True, width_mult_list=self._width_mult_list)

        self.ffm1 = FeatureFusion(self.num_filters(8, self._stem_head_width[1]), self.num_filters(8, self._stem_head_width[1]), reduction=1, Fch=self._Fch, scale=8, branch=1, norm_layer=BatchNorm2d)
        self.ffm2 = FeatureFusion(self.num_filters(8, self._stem_head_width[1]) * 2, self.num_filters(8, self._stem_head_width[1]) * 2, reduction=1, Fch=self._Fch, scale=8, branch=2, norm_layer=BatchNorm2d)
        self.ffm3 = FeatureFusion(self.num_filters(8, self._stem_head_width[1]) * 3, self.num_filters(8, self._stem_head_width[1]) * 3, reduction=1, Fch=self._Fch, scale=8, branch=3, norm_layer=BatchNorm2d)

    def agg_ffm(self, outputs8, outputs16, outputs32):
        pred32 = []; pred16 = []; pred8 = [] # order of predictions is not important
        for branch in range(self.branch_num):
            last = self.lasts[branch]
            if last == 2:
                if self.training: pred32.append(outputs32[branch])
                out = self.arms32[0](outputs32[branch])
                out = F.interpolate(out, size=(int(out.size(2))*2, int(out.size(3))*2), mode='bilinear', align_corners=True)
                out = self.refines32[0](torch.cat([out, outputs16[branch]], dim=1))
                if self.training: pred16.append(self.refines16_add(outputs16[branch]))
                out = self.arms32[1](out)
                out = F.interpolate(out, size=(int(out.size(2))*2, int(out.size(3))*2), mode='bilinear', align_corners=True)
                out = self.refines32[1](torch.cat([out, outputs8[branch]], dim=1))
                pred8.append(out)
            elif last == 1:
                if self.training: pred16.append(outputs16[branch])
                out = self.arms16(outputs16[branch])
                out = F.interpolate(out, size=(int(out.size(2))*2, int(out.size(3))*2), mode='bilinear', align_corners=True)
                out = self.refines16(torch.cat([out, outputs8[branch]], dim=1))
                pred8.append(out)
            elif last == 0:
                pred8.append(outputs8[branch])

        if len(pred32) == 0:
            pred32 = None
        else:
            pred32 = self.heads32(torch.cat(pred32, dim=1))

        if len(pred16) == 0:
            pred16 = None
        elif len(pred16) == 1:
            pred16 = self.heads16_1(torch.cat(pred16, dim=1))
        else:
            pred16 = self.heads16_2(torch.cat(pred16, dim=1))

        if len(pred8) == 0:
            pred8 = None
        elif len(pred8) == 1:
            pred8 = self.heads8_1(self.ffm1(torch.cat(pred8, dim=1)))
        elif len(pred8) == 2:
            pred8 = self.heads8_2(self.ffm2(torch.cat(pred8, dim=1)))
        else:
            pred8 = self.heads8_3(self.ffm3(torch.cat(pred8, dim=1)))

        if self.training: 
            return pred8, pred16, pred32
        else:
            if pred8 is not None:
                return pred8
            if pred16 is not None:
                return pred16
            if pred32 is not None:
                return pred32

    def ea_decode(self, seeds, path_var=33, path_index=[0, 1, 0]):

        self.lasts = [] # [0, 1]
        self.downs = [] # 0 keep, 1 down 
        self.cands = []
        self.ratios = []
        # decoded archs
        self.ops_compact = []
        self.paths_compact = []
        self.downs_compact = []
        self.widths_compact = []

        path_index = np.array(path_index)
        self.branch_num = sum(path_index)

        if self.branch_num == 0:
            return False
        elif self.branch_num == 1:
            locate = np.where(path_index>0)[0][0]
            seeds = seeds[locate*path_var:(locate+1)*path_var]
        elif self.branch_num == 2:
            locate1 = np.where(path_index>0)[0][0]
            locate2 = np.where(path_index>0)[0][1]
            seeds = np.concatenate((seeds[locate1*path_var:(locate1+1)*path_var], seeds[locate2*path_var:(locate2+1)*path_var]))
        else:
            pass

        for i in range(self.branch_num):
            downs = np.zeros(self._layers, dtype=int)
            downs_index = seeds[i*path_var : i*path_var+2]
            cands = seeds[i*path_var+2 : i*path_var+2+self._layers]
            widths = seeds[i*path_var+2+self._layers : (i+1)*path_var]
            ratios = [self._width_mult_list[width_index] for width_index in widths]

            if downs_index[0] == downs_index[1] and downs_index[0] >= 0:
                return False
                # downs_index = np.random.choice(range(0, self._layers-1), 2, replace=False)

            for down_index in downs_index:
                if down_index >= 0:
                    downs[down_index] = 1
                if cands[down_index] == 0:
                    # cands[down_index] = 1
                    return False

            self.lasts.append(sum(downs))
            ops_compact, path_compact, downs_compact, widths_compact = self.decode_arch(downs, cands, ratios)

            if downs_compact[-1] == 1:
                return False

            self.downs.append(downs)
            self.cands.append(cands)
            self.ratios.append(ratios)
            self.ops_compact.append(ops_compact)
            self.paths_compact.append(path_compact)
            self.downs_compact.append(downs_compact)
            self.widths_compact.append(widths_compact)
        
        if self.branch_num == 2:
            if self.lasts[0] == self.lasts[1]:
                return False
        elif self.branch_num == 3:
            if set(self.lasts) == set([0,1,2]):
                pass
            else:
                return False
        else:
            pass
        return True

    def decode_arch(self, downs, cands, ratios):
        ops = []
        downs_compact = []
        path_compact = []
        widths_compact = []

        path = [sum(downs[:idx]) for idx in range(self._layers)]
        for i in range(self._layers):
            if cands[i] == 0:
                if i == self._layers - 1:
                    widths_compact.pop()
                continue
            else:
                ops.append(cands[i])
                downs_compact.append(downs[i])
                path_compact.append(path[i])
                if i < (self._layers - 1):
                    widths_compact.append(ratios[i])
        assert len(path_compact) == len(widths_compact) + 1 or len(path_compact) + len(widths_compact) == 0, "path %d, widt0 %d"%(len(path_compact), len(widths_compact))
        return ops, path_compact, downs_compact, widths_compact

    def random_sample_1path(self, last=2, channel_constrain=False):
        downs = np.zeros(self._layers, dtype=int)
        cands = np.zeros(self._layers, dtype=int)
        ratios = np.ones(self._layers-1, dtype=float)
        
        if last>0:
            down_index = np.random.choice(range(0, self._layers-1), last, replace=False)
            downs[down_index] = 1

        for i in range(self._layers):
            if downs[i] or (i == self._layers-1):
                cands[i] = np.random.randint(1, len(PRIMITIVES))
            else:
                p = np.random.rand()
                if p > self.sk_ratio:
                    # cands[i] = np.random.randint(0, len(PRIMITIVES))
                    cands[i] = np.random.randint(1, len(PRIMITIVES))
                else:
                    cands[i] = 0
        
        index_l = 0
        index_h = len(self._width_mult_list)
        for j in range(self._layers - 1):
            width_index = np.random.randint(index_l, index_h)
            if channel_constrain:
                index_l = max(index_l, width_index)
            ratios[j] = self._width_mult_list[width_index]

        return downs, cands, ratios
    
    def sk_change(self, rand_int=0, sk_ration=0.5, sk_jump=0.3):
        if rand_int % 3 == 0:
            self.sk_ratio = max(0., sk_ration + sk_jump)
            # self.sk_ratio = sk_ration
        elif rand_int % 3 == 1:
            self.sk_ratio = sk_ration
            # self.sk_ratio = max(0., sk_ration - sk_jump)
        else:
            self.sk_ratio = min(1.0, sk_ration - sk_jump)

    def sample_arch(self, branch_num=None, iteration=0, channel_constrain=False):
        # keep the sample path has the same ratio
        if branch_num is None:
            self.branch_num = np.random.randint(1, 4)
        else:
            self.branch_num = branch_num

        if self.branch_num == 1:
            self.lasts = [np.random.randint(0, 3)]
        elif self.branch_num == 2:
            self.lasts = np.random.choice([0, 1, 2], 2, replace=False).tolist()
        elif self.branch_num ==3:
            self.lasts = [0, 1, 2]
        else:
            raise Exception("Not support!")

        self.downs = [] 
        self.cands = []
        self.ratios = []
        self.ops_compact = []
        self.paths_compact = []
        self.downs_compact = []
        self.widths_compact = []

        for branch_idx in range(self.branch_num):
            # self.sk_change(np.random.randint(3), self.sk_ratio_balance, self.sk_jump)
            self.sk_change(branch_idx+(3-self.branch_num)%3, self.sk_ratio_balance, self.sk_jump)
            downs, cands, ratios = self.random_sample_1path(self.lasts[branch_idx], channel_constrain=channel_constrain)
            ops_compact, path_compact, downs_compact, widths_compact = self.decode_arch(downs, cands, ratios)

            self.downs.append(downs)
            self.cands.append(cands)
            self.ratios.append(ratios)
            self.ops_compact.append(ops_compact)
            self.paths_compact.append(path_compact)
            self.downs_compact.append(downs_compact)
            self.widths_compact.append(widths_compact)

        sample_dict = {}
        sample_dict["ops"] = self.ops_compact
        sample_dict["paths"] = self.paths_compact
        sample_dict["downs"] = self.downs_compact
        sample_dict["widths"] = self.widths_compact
        sample_dict["lasts"] = self.lasts
        return sample_dict

    def forward(self, input):
        # out_prev: cell-state
        # index 0: keep; index 1: down
        input_shape = input.shape[-2:]
        H, W = input_shape

        x_stem = self.stem(input)
        
        outputs8 = [None] * self.branch_num
        outputs16 = [None] * self.branch_num
        outputs32 = [None] * self.branch_num
        output = [None] * 3

        for branch_idx in range(self.branch_num):
            layer_idx = 0
            x = x_stem

            scale = int(H // x.size(2))
            if scale <= 8: 
                outputs8[branch_idx] = x
                if self.lasts[branch_idx] == 1:
                    self.refines16.set_ratio((self._stem_head_width[0]/2+self._stem_head_width[1]/2, self._stem_head_width[1]/2))
                if self.lasts[branch_idx] == 2:
                    self.refines32[1].set_ratio((self._stem_head_width[0]/2+self._stem_head_width[1]/2, self._stem_head_width[1]/2))

            for i in range(self._layers):
                if i == 0:
                    if self.prun_mode == 'max':
                        ratios = (self._stem_head_width[0], self._width_mult_list[-1])
                    elif self.prun_mode == 'min':
                        ratios= (self._stem_head_width[0], self._width_mult_list[0])
                    else:
                        ratios = (self._stem_head_width[0], self.ratios[branch_idx][0])
                elif i == self._layers - 1:
                    if self.prun_mode == 'max':
                        ratios = (self._width_mult_list[-1], self._stem_head_width[1])
                    elif self.prun_mode == 'min':
                        ratios = (self._width_mult_list[0], self._stem_head_width[1])
                    else:
                        ratios = (self.ratios[branch_idx][i-1], self._stem_head_width[1])
                else:
                    if self.prun_mode == 'max':
                        ratios = (self._width_mult_list[-1], self._width_mult_list[-1])
                    elif self.prun_mode == 'min':
                        ratios = (self._width_mult_list[0], self._width_mult_list[0])
                    else:
                        ratios= (self.ratios[branch_idx][i-1], self.ratios[branch_idx][i])

                cell = self.cells[i][layer_idx]
                if self.sepbn:
                    x = cell(x, self.cands[branch_idx][i], ratios, self.downs[branch_idx][i], self.branch_num-1)
                else:
                    x = cell(x, self.cands[branch_idx][i], ratios, self.downs[branch_idx][i], 0)

                scale = int(H // x.size(2))
                if scale <= 8: 
                    outputs8[branch_idx] = x
                    if self.lasts[branch_idx] == 1:
                        self.refines16.set_ratio((ratios[1]/2 + self._stem_head_width[1]/2, self._stem_head_width[1]/2))
                    if self.lasts[branch_idx] == 2:
                        self.refines32[1].set_ratio((ratios[1]/2 + self._stem_head_width[1]/2, self._stem_head_width[1]/2))
                elif scale <= 16: 
                    outputs16[branch_idx] = x                        
                    if self.lasts[branch_idx] == 2:
                        self.refines16_add.set_ratio((ratios[1], self._stem_head_width[1]))
                        self.refines32[0].set_ratio((ratios[1]/2 + self._stem_head_width[1]/2, self._stem_head_width[1]/2))
                elif scale <= 32: 
                    outputs32[branch_idx] = x

                if self.downs[branch_idx][i]:
                    layer_idx = layer_idx + 1

            output[self.lasts[branch_idx]] = x

        # outputs8 = [output[0]]; outputs16=[output[1]]; outputs32 = [output[2]]

        if self.training:
            pred8, pred16, pred32 = self.agg_ffm(outputs8, outputs16, outputs32)
            if self.upsampling:
                pred8 = F.interpolate(pred8, scale_factor=8, mode='bilinear', align_corners=True)
                if pred16 is not None: pred16 = F.interpolate(pred16, scale_factor=16, mode='bilinear', align_corners=True)
                if pred32 is not None: pred32 = F.interpolate(pred32, scale_factor=32, mode='bilinear', align_corners=True)
            else:
                if pred16 is not None: pred16 = F.interpolate(pred16, scale_factor=2, mode='bilinear', align_corners=True)
                if pred32 is not None: pred32 = F.interpolate(pred32, scale_factor=4, mode='bilinear', align_corners=True)
            
            if pred32 is None and pred16 is None:
                return [pred8]
            if pred32 is None:
                return [pred8, pred16]
            return [pred8, pred16, pred32]
        else:
            pred8 = self.agg_ffm(outputs8, outputs16, outputs32)
            out = F.interpolate(pred8, size=(int(H), int(W)), mode='bilinear', align_corners=True)
            return out


    def forward_latency(self, size):

        # out_prev: cell-state
        # index 0: keep; index 1: down

        stem = self.stem
        stem_latency = 0
        total_latency = 0
        latency, size = stem[0].forward_latency(size); stem_latency = stem_latency + latency
        latency, size = stem[1].forward_latency(size); stem_latency = stem_latency + latency
        latency, size = stem[2].forward_latency(size); stem_latency = stem_latency + latency

        total_latency = total_latency + stem_latency 

        layer_idx1 = 0 # 0, 1, 2
        layer_idx2 = 0 # 0, 1, 2
        output = [None, None, None]

        size1 = size
        size2 = size 

        first_flag = True
        for i in range(self._layers):
            for branch_idx in range(self.branch_num):
                if i == 0:
                    if self.prun_mode == 'max':
                        ratios = (self._stem_head_width[0], self._width_mult_list[-1])
                    elif self.prun_mode == 'min':
                        ratios = (self._stem_head_width[0], self._width_mult_list[0])
                    else:
                        ratios = (self._stem_head_width[0], self._width_mult_list[self.ratios[branch_idx][0]])
                elif i == self._layers - 1:
                    if self.prun_mode == 'max':
                        ratios = (self._width_mult_list[-1], self._stem_head_width[1])
                    elif self.prun_mode == 'min':
                        ratios = (self._width_mult_list[0], self._stem_head_width[1])
                    else:
                        ratios = (self._width_mult_list[self.ratios[branch_idx][i]], self._stem_head_width[1])
                else:
                    if self.prun_mode == 'max':
                        ratios = (self._width_mult_list[-1], self._width_mult_list[-1])
                    elif self.prun_mode == 'min':
                        ratios = (self._width_mult_list[0], self._width_mult_list[0])
                    else:
                        ratios = (self._width_mult_list[self.ratios[branch_idx][i-1]], self._width_mult_list[self.ratios[branch_idx][i]])

            if False and first_flag:
                size = size1
                cell = self.cells[i][layer_idx1]
                latency, size = cell.forward_latency(size, self.cands[branch_idx][i], ratios, self.downs[branch_idx][i])
                size1 = size
                size2 = size
                total_latency = total_latency + latency
            else:
                first_flag = False
                cell1 = self.cells[i][layer_idx1]
                cell2 = self.cells[i][layer_idx2]
                latency1, size1 = cell1.forward_latency(size1)
                latency2, size2 = cell2.forward_latency(size2)
                total_latency = total_latency + latency1 + latency2

            if self.downs[0][i]:
                layer_idx1 = layer_idx1 + 1
            if self.downs[1][i]:
                layer_idx2 = layer_idx2 + 1
        
        output[self.lasts[0]] = size1
        output[self.lasts[1]] = size2

        out0 = output[0]; out1=output[1]; out2 = output[2]

        if out0 is not None:
            latency, size2 = self.refine32[0].forward_latency(output[2])
            size2[1:] = 2 * size2[1:]
            total_latency = total_latency + latency
            latency, size2 = self.refine32[1].forward_latency(size2)
            size2[1:] = 2 * size2[1:]
            total_latency = total_latency + latency

            latency, _ = self.head0.forward_latency(out0)
            total_latency = total_latency + latency
            latency, _ = self.head2.forward_latency(size2)
            total_latency = total_latency + latency
            size2[0] = 2 * size2[0]
            latency, _ = self.head02.forward_latency(size2)
            total_latency = total_latency + latency

        elif out1 is not None:
            latency, size1 = self.refine32[0].forward_latency(output[1])
            size1[1:] = 2 * size1[1:]
            total_latency = total_latency + latency

            latency, size2 = self.refine32[0].forward_latency(output[2])
            size2[1:] = 2 * size2[1:]
            total_latency = total_latency + latency
            latency, size2 = self.refine32[1].forward_latency(size2)
            size2[1:] = 2 * size2[1:]
            total_latency = total_latency + latency

            latency, _ = self.head1.forward_latency(size1)
            total_latency = total_latency + latency
            latency, _ = self.head2.forward_latency(size2)
            total_latency = total_latency + latency
            size2[0] = 2 * size2[0]
            latency, _ = self.head12.forward_latency(size2)
            total_latency = total_latency + latency
        else:
            raise Exception("Not support!")

        return total_latency

    def _loss(self, input, target):
        loss = 0
        self.prun_mode = None
        logits = self(input)
        raw_loss = sum(self._criterion(logit, target) for logit in logits)
        loss = loss + raw_loss
        if len(self._width_mult_list) > 1:
            self.prun_mode = "max"
            logits = self(input)
            loss = loss + sum(self._criterion(logit, target) for logit in logits)
            self.prun_mode = "min"
            logits = self(input)
            loss = loss + sum(self._criterion(logit, target) for logit in logits)
        return loss

    def setbn_loss(self, input, target):
        loss = 0
        self.prun_mode = None
        logits = self(input)
        raw_loss = sum(self._criterion(logit, target) for logit in logits)
        loss = loss + raw_loss
        return loss
