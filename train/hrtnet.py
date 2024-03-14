__all__ = ['HyCTAS', 'HRTInsHead1', 'HRTInsHead2', 'HRTInsHeadLight', 'HRTSemHead1', 'HRTSemHead2', 'HRTSemHeadLight'
]

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
from detectron2.layers import Conv2d, get_norm
import fvcore.nn.weight_init as weight_init

from operations import *
from operations import DropPath_
from genotypes import PRIMITIVES

def softmax(x):
    return np.exp(x) / (np.exp(x).sum() + np.spacing(1))

def path2downs(path):
    '''
    0 same 1 down
    '''
    downs = []
    prev = path[0]
    for node in path[1:]:
        assert (node - prev) in [0, 1]
        if node > prev:
            downs.append(1)
        else:
            downs.append(0)
        prev = node
    downs.append(0)
    return downs

def downs2path(downs):
    path = [0]
    for down in downs[:-1]:
        if down == 0:
            path.append(path[-1])
        elif down == 1:
            path.append(path[-1]+1)
    return path

def alphas2ops_path_width(alphas, path, widths, ignore_skip=False):
    '''
    alphas: [alphas0, ..., alphas3]
    '''
    assert len(path) == len(widths) + 1, "len(path) %d, len(widths) %d"%(len(path), len(widths))
    ops = []
    path_compact = []
    widths_compact = []
    pos2alpha_skips = [] # (pos, alpha of skip) to be prunned
    min_len = int(np.round(len(path) / 3.)) + path[-1] * 2
    # keep record of position(s) of skip_connect
    for i in range(len(path)):
        scale = path[i]
        if ignore_skip:
            alphas[scale][i-scale][0] = -float('inf')
        op = alphas[scale][i-scale].argmax()
        if op == 0 and (i == len(path)-1 or path[i] == path[i+1]):
            # alpha not softmax yet
            pos2alpha_skips.append((i, F.softmax(alphas[scale][i-scale], dim=-1)[0]))

    pos_skips = [ pos for pos, alpha in pos2alpha_skips ]
    pos_downs = [ pos for pos in range(len(path)-1) if path[pos] < path[pos+1] ]
    if len(pos_downs) > 0:
        pos_downs.append(len(path))
        for i in range(len(pos_downs)-1):
            # cannot be all skip_connect between each downsample-pair
            # including the last down to the path-end
            pos1 = pos_downs[i]; pos2 = pos_downs[i+1]
            if pos1+1 in pos_skips and pos2-1 in pos_skips and pos_skips.index(pos2-1) - pos_skips.index(pos1+1) == (pos2-1) - (pos1+1):
                min_skip = [1, -1] # score, pos
                for j in range(pos1+1, pos2):
                    scale = path[j]
                    score = F.softmax(alphas[scale][j-scale], dim=-1)[0]
                    if score <= min_skip[0]:
                        min_skip = [score, j]
                alphas[path[min_skip[1]]][min_skip[1]-path[min_skip[1]]][0] = -float('inf')

    if len(pos2alpha_skips) > len(path) - min_len:
        pos2alpha_skips = sorted(pos2alpha_skips, key=lambda x: x[1], reverse=True)[:len(path) - min_len]
    pos_skips = [ pos for pos, alpha in pos2alpha_skips ]
    for i in range(len(path)):
        scale = path[i]
        if i < len(widths): width = widths[i]
        op = alphas[scale][i-scale].argmax()
        if op == 0:
            if i in pos_skips:
                # remove the last width if the last layer (skip_connect) is to be prunned
                if i == len(path) - 1: widths_compact = widths_compact[:-1]
                continue
            else:
                alphas[scale][i-scale][0] = -float('inf')
                op = alphas[scale][i-scale].argmax()
        path_compact.append(scale)
        if i < len(widths): widths_compact.append(width)
        ops.append(op)
    assert len(path_compact) >= min_len
    return ops, path_compact, widths_compact

def betas2path(betas, last, layers):
    downs = [0] * layers
    # betas1 is of length layers-2; beta2: layers-3; beta3: layers-4
    if last == 1:
        down_idx = np.argmax([ beta[0] for beta in betas[1][1:-1].cpu().numpy() ]) + 1
        downs[down_idx] = 1
    elif last == 2:
        max_prob = 0; max_ij = (0, 1)
        for j in range(layers-4):
            for i in range(1, j-1):
                prob = betas[1][i][0] * betas[2][j][0]
                if prob > max_prob:
                    max_ij = (i, j)
                    max_prob = prob
        downs[max_ij[0]+1] = 1; downs[max_ij[1]+2] = 1
    path = downs2path(downs)
    assert path[-1] == last
    return path

def path2widths(path, ratios, width_mult_list):
    widths = []
    for layer in range(1, len(path)):
        scale = path[layer]
        if scale == 0:
            widths.append(width_mult_list[ratios[scale][layer-1].argmax()])
        else:
            widths.append(width_mult_list[ratios[scale][layer-scale].argmax()])
    return widths

def network_metas(alphas, betas, ratios, width_mult_list, layers, last, ignore_skip=False):
    betas[1] = F.softmax(betas[1], dim=-1)
    betas[2] = F.softmax(betas[2], dim=-1)
    path = betas2path(betas, last, layers)
    widths = path2widths(path, ratios, width_mult_list)
    ops, path, widths = alphas2ops_path_width(alphas, path, widths, ignore_skip=ignore_skip)
    assert len(ops) == len(path) and len(path) == len(widths) + 1, "op %d, path %d, width%d"%(len(ops), len(path), len(widths))
    downs = path2downs(path) # 0 same 1 down
    return ops, path, downs, widths
    
class MixedOp(nn.Module):
    def __init__(self, norm, C_in, C_out, op_idx, stride=1, use_bias=False, fmap_size=(64, 128)):
        super(MixedOp, self).__init__()
        op = OPS[PRIMITIVES[op_idx]](norm, C_in, C_out, stride, use_bias, fmap_size=fmap_size)
        self.sk = True
        if not isinstance(op, FactorizedReduce): # Identity does not use drop path
            self.sk = False
            op = nn.Sequential(
                op,
                DropPath_()
            )
        self._op = op

    def forward(self, x):
        return self._op(x)

class Cell(nn.Module):
    def __init__(self, norm, op_idx, C_in, C_out, down, use_bias, fmap_size):
        super(Cell, self).__init__()
        self._C_in = C_in
        self._C_out = C_out
        self._down = down

        if self._down:
            self._op = MixedOp(norm, C_in, C_out, op_idx, stride=2, use_bias=use_bias, fmap_size=fmap_size)
        else:
            self._op = MixedOp(norm, C_in, C_out, op_idx, use_bias=use_bias, fmap_size=fmap_size)

    def forward(self, input):
        out = self._op(input)
        return out


class FeatureFusion(nn.Module):
    def __init__(self, norm, in_planes, out_planes, reduction=1, Fch=16, scale=4, branch=2, use_bias=False):
        super(FeatureFusion, self).__init__()
        self.conv_1x1 = Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=use_bias,
                norm=get_norm(norm, out_planes),
                activation=F.relu,
            )
        weight_init.c2_xavier_fill(self.conv_1x1)
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2d(
                out_planes,
                out_planes // reduction,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=use_bias,
                norm=get_norm(norm, out_planes // reduction),
                activation=F.relu,
            ),
            Conv2d(
                out_planes // reduction,
                out_planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=use_bias,
                norm=get_norm(norm, out_planes)
            ),
            nn.Sigmoid()
        )
        weight_init.c2_xavier_fill(self.channel_attention[1])
        weight_init.c2_xavier_fill(self.channel_attention[2])
        self._Fch = Fch
        self._scale = scale
        self._branch = branch

    def forward(self, fm):
        # fm is already a concatenation of multiple scales
        fm = self.conv_1x1(fm)
        # return fm
        fm_se = self.channel_attention(fm)
        output = fm + fm * fm_se
        return output

class HyCTAS(nn.Module):
    def __init__(self, ops, paths, downs, widths, lasts, norm, layers=20, Fch=16, width_mult_list=[1.,], stem_head_width=(1., 1.), use_bias=False, align_corners=False, input_size=(1024, 2048)):
        super(HyCTAS, self).__init__()
        assert layers >= 2
        self._layers = layers
        self._Fch = Fch
        self._width_mult_list = width_mult_list
        self._stem_head_width = stem_head_width
        self.latency = 0
        self.use_bias = use_bias
        self.input_size = input_size
        self.norm = norm
        self.align_corners = align_corners
        self.outs16_inplanes = None
        self.outs32_inplanes = None

        self.stem1 = nn.Sequential(
            Conv2d(
                3,
                self.num_filters(2, stem_head_width[0])*2,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=1,
                bias=False,
                norm=get_norm(norm, self.num_filters(2, stem_head_width[0])*2)
            ),
            BasicResidual2x(norm, self.num_filters(2, stem_head_width[0])*2, self.num_filters(4, stem_head_width[0])*2, kernel_size=3, stride=2, groups=1, use_bias=use_bias)
        )
        weight_init.c2_xavier_fill(self.stem1[0])

        self.stem2 = BasicResidual2x(norm, self.num_filters(4, stem_head_width[0])*2, self.num_filters(8, stem_head_width[0]), kernel_size=3, stride=2, groups=1, use_bias=use_bias)


        self.ops = ops
        self.paths = paths
        self.downs = downs
        self.widths = widths
        self._branch = len(lasts)
        self.lasts = lasts
        self.branch_groups, self.cells = self.get_branch_groups_cells(self.ops, self.paths, self.downs, self.widths, self.lasts)
        self.build_arm_ffm_head()
        self.shape8 = [None, None]
        self.shape16 = [None, None]

        self.s4_inplanes = self.num_filters(4, stem_head_width[0])*2
        self.s8_inplanes = self.num_filters(8, stem_head_width[0])
        self.outs8_inplanes = self.num_filters(8, self._stem_head_width[1]) * self._branch



    def num_filters(self, scale, width=1.0):
        return int(np.round(scale * self._Fch * width))
    
    def drop_path_prob(self, p):
        """ Set drop path probability """
        for module in self.modules():
            if isinstance(module, DropPath_):
                module.p = p
                
    def build_arm_ffm_head(self):
        self.ffm16 = None
        self.ffm32 = None
        if 2 in self.lasts:
            self.ffm32 = FeatureFusion(self.norm, self.num_filters(32, self._stem_head_width[1]), self.num_filters(32, self._stem_head_width[1]), reduction=1, Fch=self._Fch, scale=32, branch=self._branch, use_bias=self.use_bias)
            if 1 in self.lasts:
                self.ffm16 = FeatureFusion(self.norm, self.num_filters(16, self._stem_head_width[1])+self.ch_16, self.num_filters(16, self._stem_head_width[1]), reduction=1, Fch=self._Fch, scale=16, branch=self._branch, use_bias=self.use_bias)
            else:
                self.ffm16 = FeatureFusion(self.norm, self.ch_16, self.num_filters(16, self._stem_head_width[1]), reduction=1, Fch=self._Fch, scale=16, branch=self._branch, use_bias=self.use_bias)

            self.outs32_inplanes = self.num_filters(32, self._stem_head_width[1])
            self.outs16_inplanes = self.num_filters(16, self._stem_head_width[1])
        else:
            if 1 in self.lasts:
                self.ffm16 = FeatureFusion(self.norm, self.num_filters(16, self._stem_head_width[1]), self.num_filters(16, self._stem_head_width[1]), reduction=1, Fch=self._Fch, scale=16, branch=self._branch, use_bias=self.use_bias)
                self.outs16_inplanes = self.num_filters(16, self._stem_head_width[1])
            
        if 2 in self.lasts:
            self.arms32 = nn.ModuleList([
                Conv2d(
                    self.num_filters(32, self._stem_head_width[1]),
                    self.num_filters(16, self._stem_head_width[1]),
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=self.use_bias,
                    norm=get_norm(self.norm, self.num_filters(16, self._stem_head_width[1]))
                ),
                Conv2d(
                    self.num_filters(16, self._stem_head_width[1]),
                    self.num_filters(8, self._stem_head_width[1]),
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=self.use_bias,
                    norm=get_norm(self.norm, self.num_filters(8, self._stem_head_width[1]))
                )
            ])
            weight_init.c2_xavier_fill(self.arms32[0])
            weight_init.c2_xavier_fill(self.arms32[1])

            self.refines32 = nn.ModuleList([
                Conv2d(
                    self.num_filters(16, self._stem_head_width[1])+self.ch_16,
                    self.num_filters(16, self._stem_head_width[1]),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=self.use_bias,
                    norm=get_norm(self.norm, self.num_filters(16, self._stem_head_width[1]))
                ),
                Conv2d(
                    self.num_filters(8, self._stem_head_width[1])+self.ch_8_2,
                    self.num_filters(8, self._stem_head_width[1]),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=self.use_bias,
                    norm=get_norm(self.norm, self.num_filters(8, self._stem_head_width[1]))
                )
            ])
            weight_init.c2_xavier_fill(self.refines32[0])
            weight_init.c2_xavier_fill(self.refines32[0])

        if 1 in self.lasts:
            self.arms16 = Conv2d(
                    self.num_filters(16, self._stem_head_width[1]),
                    self.num_filters(8, self._stem_head_width[1]),
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=self.use_bias,
                    norm=get_norm(self.norm, self.num_filters(8, self._stem_head_width[1]))
                )
            weight_init.c2_xavier_fill(self.arms16)
            
            self.refines16 = Conv2d(
                    self.num_filters(8, self._stem_head_width[1])+self.ch_8_1,
                    self.num_filters(8, self._stem_head_width[1]),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=self.use_bias,
                    norm=get_norm(self.norm, self.num_filters(8, self._stem_head_width[1]))
                )
            weight_init.c2_xavier_fill(self.refines16)
            
        self.ffm = FeatureFusion(self.norm, self.num_filters(8, self._stem_head_width[1]) * self._branch, self.num_filters(8, self._stem_head_width[1]) * self._branch, reduction=1, Fch=self._Fch, scale=8, branch=self._branch, use_bias=self.use_bias)

    def get_branch_groups_cells(self, ops, paths, downs, widths, lasts):
        num_branch = len(ops)
        layers = max([len(path) for path in paths])
        groups_all = []
        self.ch_16 = 0; self.ch_8_2 = 0; self.ch_8_1 = 0
        cells = nn.ModuleDict() # layer-branch: op
        branch_connections = np.ones((num_branch, num_branch)) # maintain connections of heads of branches of different scales
        # all but the last layer
        # we determine branch-merging by comparing their next layer: if next-layer differs, then the "down" of current layer must differ
        for l in range(layers):
            connections = np.ones((num_branch, num_branch)) # if branch i/j share same scale & op in this layer
            for i in range(num_branch):
                for j in range(i+1, num_branch):
                    # we also add constraint on ops[i][l] != ops[j][l] since some skip-connect may already be shrinked/compacted => layers of branches may no longer aligned in terms of alphas
                    # last layer won't merge
                    if len(paths[i]) <= l+1 or len(paths[j]) <= l+1 or paths[i][l+1] != paths[j][l+1] or ops[i][l] != ops[j][l] or widths[i][l] != widths[j][l]:
                        connections[i, j] = connections[j, i] = 0
            branch_connections *= connections
            branch_groups = []
            # build branch_group for processing
            for branch in range(num_branch):
                # also accept if this is the last layer of branch (len(paths[branch]) == l+1)
                if len(paths[branch]) < l+1: continue
                inserted = False
                for group in branch_groups:
                    if branch_connections[group[0], branch] == 1:
                        group.append(branch)
                        inserted = True
                        continue
                if not inserted:
                    branch_groups.append([branch])
            for group in branch_groups:
                # branch in the same group must share the same op/scale/down/width
                if len(group) >= 2: assert ops[group[0]][l] == ops[group[1]][l] and paths[group[0]][l+1] == paths[group[1]][l+1] and downs[group[0]][l] == downs[group[1]][l] and widths[group[0]][l] == widths[group[1]][l]
                if len(group) == 3: assert ops[group[1]][l] == ops[group[2]][l] and paths[group[1]][l+1] == paths[group[2]][l+1] and downs[group[1]][l] == downs[group[2]][l] and widths[group[1]][l] == widths[group[2]][l]
                op = ops[group[0]][l]
                scale = 2**(paths[group[0]][l]+3)
                down = downs[group[0]][l]
                if l < len(paths[group[0]]) - 1: assert down == paths[group[0]][l+1] - paths[group[0]][l]
                assert down in [0, 1]
                if l == 0:
                    cell = Cell(self.norm, op, self.num_filters(scale, self._stem_head_width[0]), self.num_filters(scale*(down+1), widths[group[0]][l]), down, use_bias=self.use_bias, fmap_size=(self.input_size[0] // scale, self.input_size[1] // scale))
                elif l == len(paths[group[0]]) - 1:
                    # last cell for this branch
                    assert down == 0
                    cell = Cell(self.norm, op, self.num_filters(scale, widths[group[0]][l-1]), self.num_filters(scale, self._stem_head_width[1]), down, use_bias=self.use_bias, fmap_size=(self.input_size[0] // scale, self.input_size[1] // scale))
                else:
                    cell = Cell(self.norm, op, self.num_filters(scale, widths[group[0]][l-1]), self.num_filters(scale*(down+1), widths[group[0]][l]), down, use_bias=self.use_bias, fmap_size=(self.input_size[0] // scale, self.input_size[1] // scale))
                # For Feature Fusion: keep record of dynamic #channel of last 1/16 and 1/8 of "1/32 branch"; last 1/8 of "1/16 branch"
                if 2 in self.lasts and self.lasts.index(2) in group and down and scale == 16: self.ch_16 = cell._C_in
                if 2 in self.lasts and self.lasts.index(2) in group and down and scale == 8: self.ch_8_2 = cell._C_in
                if 1 in self.lasts and self.lasts.index(1) in group and down and scale == 8: self.ch_8_1 = cell._C_in
                for branch in group:
                    cells[str(l)+"-"+str(branch)] = cell
            groups_all.append(branch_groups)
        return groups_all, cells
    

    def agg_ffm(self, outputs8, outputs16, outputs32):
        pred32 = []; pred16 = []; pred8 = [] # order of predictions is not important
        for branch in range(self._branch):
            last = self.lasts[branch]
            if last == 2:
                pred32.append(outputs32[branch])
                out = self.arms32[0](outputs32[branch])
                out = F.interpolate(out, size=(self.shape16[0], self.shape16[1]), mode='bilinear', align_corners=self.align_corners)
                out = self.refines32[0](torch.cat([out, outputs16[branch]], dim=1))
                pred16.append(outputs16[branch])
                out = self.arms32[1](out)
                out = F.interpolate(out, size=(self.shape8[0], self.shape8[1]), mode='bilinear', align_corners=self.align_corners)
                out = self.refines32[1](torch.cat([out, outputs8[branch]], dim=1))
                pred8.append(out)
            elif last == 1:
                pred16.append(outputs16[branch])
                out = self.arms16(outputs16[branch])
                out = F.interpolate(out, size=(self.shape8[0], self.shape8[1]), mode='bilinear', align_corners=self.align_corners)
                out = self.refines16(torch.cat([out, outputs8[branch]], dim=1))
                pred8.append(out)
            elif last == 0:
                pred8.append(outputs8[branch])
        
        s8_feat = self.ffm(torch.cat(pred8, dim=1))
        s16_feat = None
        if self.ffm16 is not None:
            s16_feat = self.ffm16(torch.cat(pred16, dim=1))
        s32_feat = None
        if self.ffm32 is not None:
            s32_feat = self.ffm32(torch.cat(pred32, dim=1))

        return s8_feat, s16_feat, s32_feat
        # return s8_feat, s8_feat, s8_feat

    def forward(self, input):
        input_shape = input.shape[-2:]
        H, W = input_shape

        stem = self.stem1(input)
        low_level_feature_s4 = stem
        stem = self.stem2(stem)
        low_level_feature_s8 = stem

        # store the last feature map w. corresponding scale of each branch
        outputs8 = [stem] * self._branch
        outputs16 = [stem] * self._branch
        outputs32 = [stem] * self._branch
        outputs = [stem] * self._branch

        for layer in range(len(self.branch_groups)):
            for group in self.branch_groups[layer]:
                output = self.cells[str(layer)+"-"+str(group[0])](outputs[group[0]])
                scale = int(H // output.size(2))
                for branch in group:
                    outputs[branch] = output
                    if scale <= 8:
                         outputs8[branch] = output
                         self.shape8 = output.shape[2:]
                    elif scale <= 16:
                         outputs16[branch] = output
                         self.shape16 = output.shape[2:]
                    elif scale <= 32: outputs32[branch] = output
        
        s8_feat, s16_feat, s32_feat = self.agg_ffm(outputs8, outputs16, outputs32)

        return low_level_feature_s4, low_level_feature_s8, s8_feat, s16_feat, s32_feat

class HRTInsHeadLight(nn.Module):
    def __init__(self, norm, in_planes, s4_inplanes, s8_inplanes, fmap_size=(128, 256), align_corners=False):
        super(HRTInsHeadLight, self).__init__()
        
        self.align_corners = align_corners
        if in_planes <= 64:
            mid_planes = in_planes
        elif in_planes <= 256:
            mid_planes = in_planes
        else:
            mid_planes = in_planes // 2

        decoder_planes = mid_planes // 2

        use_bias = False
        self.att_sa = SAHead(norm, dim=in_planes, fmap_size=(128, 256), dim_out=in_planes, downsample=False, align_corners=self.align_corners)
        self.center_head = BasicResidual_downup_2x(norm, in_planes, mid_planes, 3, 1, 1, use_bias=use_bias, align_corners=self.align_corners)
        self.center_conv = Conv2d(
                mid_planes,
                decoder_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1,
                bias=False,
                norm=get_norm(norm, decoder_planes),
                activation=F.relu
            )

        self.offset_head = BasicResidual_downup_2x(norm, in_planes, mid_planes, 3, 1, 1, use_bias=use_bias, align_corners=self.align_corners)
        self.offset_conv = Conv2d(
                mid_planes,
                decoder_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1,
                bias=False,
                norm=get_norm(norm, decoder_planes),
                activation=F.relu
            )

        weight_init.c2_xavier_fill(self.center_conv)
        weight_init.c2_xavier_fill(self.offset_conv)
        weight_init.c2_xavier_fill(self.center_head.conv1)
        weight_init.c2_xavier_fill(self.center_head.conv2)
        weight_init.c2_xavier_fill(self.offset_head.conv1)
        weight_init.c2_xavier_fill(self.offset_head.conv2)

        self.center_predictor = Conv2d(decoder_planes, 1, kernel_size=1)
        nn.init.normal_(self.center_predictor.weight, 0, 0.001)
        nn.init.constant_(self.center_predictor.bias, 0)

        self.offset_predictor = Conv2d(decoder_planes, 2, kernel_size=1)
        nn.init.normal_(self.offset_predictor.weight, 0, 0.001)
        nn.init.constant_(self.offset_predictor.bias, 0)
        

    def forward(self, x):

        xs = self.att_sa(x)

        # center
        center = self.center_head(xs)
        center = self.center_conv(center)
        center = self.center_predictor(center)
        # offset
        offset = self.offset_head(xs)
        offset = self.offset_conv(offset)
        offset = self.offset_predictor(offset)
        
        return center, offset

class HRTInsHead1(nn.Module):
    def __init__(self, norm, in_planes, s4_inplanes, s8_inplanes, fmap_size=(128, 256), align_corners=False):
        super(HRTInsHead1, self).__init__()

        C_low = 48
        self.align_corners = align_corners
        self.feature_projection_s4 = Conv2d(
                s4_inplanes,
                C_low,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias=False,
                norm=get_norm(norm, C_low)
            )
        
        # in_planes = in_planes + C_low
        if in_planes <= 64:
            mid_planes = in_planes
        elif in_planes <= 256:
            mid_planes = in_planes
        else:
            mid_planes = in_planes // 2

        decoder_planes = mid_planes // 2

        use_bias = False
        self.att_sa = SAHead(norm, dim=in_planes, fmap_size=(128, 256), dim_out=in_planes, downsample=False, align_corners=self.align_corners)
        self.center_head = BasicResidual_downup_2x(norm, in_planes+C_low, mid_planes, 3, 1, 1, use_bias=use_bias, align_corners=self.align_corners)
        self.center_conv = Conv2d(
                mid_planes,
                decoder_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1,
                bias=False,
                norm=get_norm(norm, decoder_planes),
                activation=F.relu
            )

        self.offset_head = BasicResidual_downup_2x(norm, in_planes+C_low, mid_planes, 3, 1, 1, use_bias=use_bias, align_corners=self.align_corners)
        self.offset_conv = Conv2d(
                mid_planes,
                decoder_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1,
                bias=False,
                norm=get_norm(norm, decoder_planes),
                activation=F.relu
            )

        weight_init.c2_xavier_fill(self.center_conv)
        weight_init.c2_xavier_fill(self.offset_conv)
        weight_init.c2_xavier_fill(self.center_head.conv1)
        weight_init.c2_xavier_fill(self.center_head.conv2)
        weight_init.c2_xavier_fill(self.offset_head.conv1)
        weight_init.c2_xavier_fill(self.offset_head.conv2)

        self.center_predictor = Conv2d(decoder_planes, 1, kernel_size=1)
        nn.init.normal_(self.center_predictor.weight, 0, 0.001)
        nn.init.constant_(self.center_predictor.bias, 0)

        self.offset_predictor = Conv2d(decoder_planes, 2, kernel_size=1)
        nn.init.normal_(self.offset_predictor.weight, 0, 0.001)
        nn.init.constant_(self.offset_predictor.bias, 0)
        

    def forward(self, features):

        s4_feat, _, x = features
        xs = self.att_sa(x)
        low_level_feat_sem = self.feature_projection_s4(s4_feat)
        xs = F.interpolate(xs, size=low_level_feat_sem.size()[2:], mode='bilinear', align_corners=self.align_corners)
        y = torch.cat((xs, low_level_feat_sem), dim=1)

        # center
        center = self.center_head(y)
        center = self.center_conv(center)
        center = self.center_predictor(center)
        # offset
        offset = self.offset_head(y)
        offset = self.offset_conv(offset)
        offset = self.offset_predictor(offset)
        
        return center, offset

class HRTInsHead2(nn.Module):
    def __init__(self, norm, in_planes, s4_inplanes, s8_inplanes, fmap_size=(128, 256), align_corners=False):
        super(HRTInsHead2, self).__init__()

        C_low1 = 32
        C_low2 = 16
        self.align_corners = align_corners

        self.feature_projection_s4 = Conv2d(
                s4_inplanes,
                C_low2,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias=False,
                norm=get_norm(norm, C_low2)
            )

        self.feature_projection_s8 = Conv2d(
                s8_inplanes,
                C_low1,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias=False,
                norm=get_norm(norm, C_low1)
            )
        # in_planes = in_planes + C_low
        if in_planes <= 64:
            mid_planes = in_planes
        elif in_planes <= 256:
            mid_planes = in_planes
        else:
            mid_planes = in_planes // 2

        decoder_planes = mid_planes // 2

        use_bias = False

        self.att_sa = SAHead(norm, dim=in_planes, fmap_size=(128, 256), dim_out=in_planes, downsample=False, align_corners=self.align_corners)

        self.decoder = BasicResidual_downup_2x(norm, in_planes+C_low1, mid_planes, 3, 1, 1, use_bias=use_bias, align_corners=self.align_corners)

        self.center_head = BasicResidual_downup_2x(norm, mid_planes+C_low2, mid_planes, 3, 1, 1, use_bias=use_bias, align_corners=self.align_corners)
        self.center_conv = Conv2d(
                mid_planes,
                decoder_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1,
                bias=False,
                norm=get_norm(norm, decoder_planes),
                activation=F.relu
            )

        self.offset_head = BasicResidual_downup_2x(norm, mid_planes+C_low2, mid_planes, 3, 1, 1, use_bias=use_bias, align_corners=self.align_corners)
        self.offset_conv = Conv2d(
                mid_planes,
                decoder_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1,
                bias=False,
                norm=get_norm(norm, decoder_planes),
                activation=F.relu
            )

        weight_init.c2_xavier_fill(self.center_conv)
        weight_init.c2_xavier_fill(self.offset_conv)
        weight_init.c2_xavier_fill(self.decoder.conv1)
        weight_init.c2_xavier_fill(self.decoder.conv2)
        weight_init.c2_xavier_fill(self.center_head.conv1)
        weight_init.c2_xavier_fill(self.center_head.conv2)
        weight_init.c2_xavier_fill(self.offset_head.conv1)
        weight_init.c2_xavier_fill(self.offset_head.conv2)

        self.center_predictor = Conv2d(decoder_planes, 1, kernel_size=1)
        nn.init.normal_(self.center_predictor.weight, 0, 0.001)
        nn.init.constant_(self.center_predictor.bias, 0)

        self.offset_predictor = Conv2d(decoder_planes, 2, kernel_size=1)
        nn.init.normal_(self.offset_predictor.weight, 0, 0.001)
        nn.init.constant_(self.offset_predictor.bias, 0)
        

    def forward(self, features):

        s4_feat, s8_feat, x = features
        xs = self.att_sa(x)
        s8_feat = self.feature_projection_s8(s8_feat)
        xs = torch.cat((xs, s8_feat), dim=1)
        xs = self.decoder(xs)

        s4_feat = self.feature_projection_s4(s4_feat)
        xs = F.interpolate(xs, size=s4_feat.size()[2:], mode='bilinear', align_corners=self.align_corners)
        xs = torch.cat((xs, s4_feat), dim=1)

        # center
        center = self.center_head(xs)
        center = self.center_conv(center)
        center = self.center_predictor(center)
        # offset
        offset = self.offset_head(xs)
        offset = self.offset_conv(offset)
        offset = self.offset_predictor(offset)
        return center, offset

class HRTSemHeadLight(nn.Module):
    def __init__(self, norm, in_planes, s4_inplanes, s8_inplanes, out_planes=19, fmap_size=(128, 256), align_corners=False):
        super(HRTSemHeadLight, self).__init__()

        self.align_corners = align_corners
        if in_planes <= 64:
            mid_planes = in_planes
        elif in_planes <= 256:
            mid_planes = in_planes
        else:
            mid_planes = in_planes // 2

        use_bias = False
        self.att_sa = SAHead(norm, dim=in_planes, fmap_size=(128, 256), dim_out=in_planes, downsample=False, align_corners=self.align_corners)
        self.seg_head = BasicResidual_downup_2x(norm, in_planes, mid_planes, 3, 1, 1, use_bias=use_bias, align_corners=self.align_corners)
        self.seg_conv = Conv2d(
                mid_planes,
                mid_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1,
                bias=False,
                norm=get_norm(norm, mid_planes),
                activation=F.relu
            )

        weight_init.c2_xavier_fill(self.seg_conv)
        weight_init.c2_xavier_fill(self.seg_head.conv1)
        weight_init.c2_xavier_fill(self.seg_head.conv2)

        self.predictor = Conv2d(mid_planes, out_planes, kernel_size=1)
        nn.init.normal_(self.predictor.weight, 0, 0.001)
        nn.init.constant_(self.predictor.bias, 0)
        
    def forward(self, x):
        xs = self.att_sa(x)
        y = self.seg_head(xs)
        y = self.seg_conv(y)
        y = self.predictor(y)
        return y

class HRTSemHead1(nn.Module):
    def __init__(self, norm, in_planes, s4_inplanes, s8_inplanes, out_planes=19, fmap_size=(128, 256), align_corners=False):
        super(HRTSemHead1, self).__init__()

        C_low = 48
        self.align_corners = align_corners
        self.feature_projection_s4 = Conv2d(
                s4_inplanes,
                C_low,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias=False,
                norm=get_norm(norm, C_low)
            )
        
        # in_planes = in_planes + C_low
        if in_planes <= 64:
            mid_planes = in_planes
        elif in_planes <= 256:
            mid_planes = in_planes
        else:
            mid_planes = in_planes // 2

        use_bias = False
        self.att_sa = SAHead(norm, dim=in_planes, fmap_size=(128, 256), dim_out=in_planes, downsample=False, align_corners=self.align_corners)
        self.seg_head = BasicResidual_downup_2x(norm, in_planes+C_low, mid_planes, 3, 1, 1, use_bias=use_bias, align_corners=self.align_corners)
        self.seg_conv = Conv2d(
                mid_planes,
                mid_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1,
                bias=False,
                norm=get_norm(norm, mid_planes),
                activation=F.relu
            )

        weight_init.c2_xavier_fill(self.seg_conv)
        weight_init.c2_xavier_fill(self.seg_head.conv1)
        weight_init.c2_xavier_fill(self.seg_head.conv2)

        self.predictor = Conv2d(mid_planes, out_planes, kernel_size=1)
        nn.init.normal_(self.predictor.weight, 0, 0.001)
        nn.init.constant_(self.predictor.bias, 0)
        

    def forward(self, features):

        s4_feat, _, x = features
        xs = self.att_sa(x)
        low_level_feat_sem = self.feature_projection_s4(s4_feat)
        xs = F.interpolate(xs, size=low_level_feat_sem.size()[2:], mode='bilinear', align_corners=self.align_corners)
        xs = torch.cat((xs, low_level_feat_sem), dim=1)

        y = self.seg_head(xs)
        y = self.seg_conv(y)
        y = self.predictor(y)
        
        return y

class HRTSemHead2(nn.Module):
    def __init__(self, norm, in_planes, s4_inplanes, s8_inplanes, out_planes=19, fmap_size=(128, 256), align_corners=False):
        super(HRTSemHead2, self).__init__()

        C_low1 = 64
        C_low2 = 32
        self.align_corners = align_corners
        self.feature_projection_s4 = Conv2d(
                s4_inplanes,
                C_low2,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias=False,
                norm=get_norm(norm, C_low2)
            )
        
        self.feature_projection_s8 = Conv2d(
                s8_inplanes,
                C_low1,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias=False,
                norm=get_norm(norm, C_low1)
            )

        # in_planes = in_planes + C_low
        if in_planes <= 64:
            mid_planes = in_planes
        elif in_planes <= 256:
            mid_planes = in_planes
        else:
            mid_planes = in_planes // 2

        use_bias = False
        self.att_sa = SAHead(norm, dim=in_planes, fmap_size=(128, 256), dim_out=in_planes, downsample=False, align_corners=self.align_corners)
        self.decoder = BasicResidual_downup_2x(norm, in_planes+C_low1, in_planes, 3, 1, 1, use_bias=use_bias, align_corners=self.align_corners)
        self.seg_head = BasicResidual_downup_2x(norm, in_planes+C_low2, mid_planes, 3, 1, 1, use_bias=use_bias, align_corners=self.align_corners)
        self.seg_conv = Conv2d(
                mid_planes,
                mid_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1,
                bias=False,
                norm=get_norm(norm, mid_planes),
                activation=F.relu
            )

        weight_init.c2_xavier_fill(self.seg_conv)
        weight_init.c2_xavier_fill(self.seg_head.conv1)
        weight_init.c2_xavier_fill(self.seg_head.conv2)

        self.predictor = Conv2d(mid_planes, out_planes, kernel_size=1)
        nn.init.normal_(self.predictor.weight, 0, 0.001)
        nn.init.constant_(self.predictor.bias, 0)
        

    def forward(self, features):

        s4_feat, s8_feat, x = features
        xs = self.att_sa(x)
        s8_feat = self.feature_projection_s8(s8_feat)
        xs = torch.cat((xs, s8_feat), dim=1)
        xs = self.decoder(xs)
        
        s4_feat = self.feature_projection_s4(s4_feat)
        xs = F.interpolate(xs, size=s4_feat.size()[2:], mode='bilinear', align_corners=self.align_corners)
        xs = torch.cat((xs, s4_feat), dim=1)

        y = self.seg_head(xs)
        y = self.seg_conv(y)
        y = self.predictor(y)
        
        return y