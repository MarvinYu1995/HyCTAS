__all__ = ['BasicResidual1x', 'BasicResidual_downup_1x', 'BasicResidual2x', 'BasicResidual_downup_2x', 'FactorizedReduce', 'OPS', 'OPS_name', 'OPS_Class', 'Self_Attn', 'SAHead']

from pdb import set_trace as bp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from thop import profile
import sys
import os.path as osp
from easydict import EasyDict as edict
from torch import nn, einsum
from einops import rearrange
import fvcore.nn.weight_init as weight_init
from detectron2.layers import Conv2d, get_norm


def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # per data point mask; assuming x in cuda.
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x = torch.div(x, keep_prob)
        x = torch.mul(x, mask)
        # x.div_(keep_prob).mul_(mask)
    return x

class DropPath_(nn.Module):
    def __init__(self, p=0.):
        """ [!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)

    def forward(self, x):
        drop_path_(x, self.p, self.training)

        return x

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        latency = 0
        return latency, (c_in, h_in, w_in)

class BasicResidual1x(nn.Module):
    def __init__(self, norm, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, use_bias=False):
        super(BasicResidual1x, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        groups = 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1

        self.conv = Conv2d(
                C_in,
                C_out,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                groups=groups,
                bias=use_bias,
                norm=get_norm(norm, C_out),
                activation=F.relu,
        )
        weight_init.c2_xavier_fill(self.conv)
    
    def forward(self, x):
        out = self.conv(x)
        return out


class BasicResidual_downup_1x(nn.Module):
    def __init__(self, norm, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, use_bias=False, align_corners=False):
        super(BasicResidual_downup_1x, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        groups = 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1
        self.align_corners = align_corners

        self.relu = F.relu
        self.conv = Conv2d(
                C_in,
                C_out,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
                groups=groups,
                bias=use_bias,
                norm=get_norm(norm, C_out)
        )
        if self.stride==1:
            self.downsample = Conv2d(
                    C_in,
                    C_out,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    dilation=dilation,
                    groups=groups,
                    bias=use_bias,
                    norm=get_norm(norm, C_out)
            )
            weight_init.c2_xavier_fill(self.downsample)

        weight_init.c2_xavier_fill(self.conv)

    def forward(self, x):
        out = F.interpolate(x, size=(int(x.size(2))//2, int(x.size(3))//2), mode='bilinear', align_corners=self.align_corners)
        out = self.conv(out)
        if self.stride == 1:
            out = F.interpolate(out, size=(int(x.size(2)), int(x.size(3))), mode='bilinear', align_corners=self.align_corners)
            out = out + self.downsample(x)
        out = self.relu(out)
        return out


class BasicResidual2x(nn.Module):
    def __init__(self, norm, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, use_bias=False):
        super(BasicResidual2x, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        groups = 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1

        self.conv1 = Conv2d(
                C_in,
                C_out,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                groups=groups,
                bias=use_bias,
                norm=get_norm(norm, C_out),
                activation=F.relu,
        )

        self.conv2 = Conv2d(
                C_out,
                C_out,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
                groups=groups,
                bias=use_bias,
                norm=get_norm(norm, C_out),
                activation=F.relu,
        )

        weight_init.c2_xavier_fill(self.conv1)
        weight_init.c2_xavier_fill(self.conv2)
        
    def forward(self, x):
        x = self.conv1(x)
        out = self.conv2(x)
        return out


class BasicResidual_downup_2x(nn.Module):
    def __init__(self, norm, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, use_bias=False, align_corners=False):
        super(BasicResidual_downup_2x, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        groups = 1
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.align_corners = align_corners
        assert stride in [1, 2]
        if self.stride == 2: self.dilation = 1

        self.relu = F.relu
        self.conv1 = Conv2d(
                C_in,
                C_out,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
                groups=groups,
                bias=use_bias,
                norm=get_norm(norm, C_out),
                activation=F.relu,
        )

        self.conv2 = Conv2d(
                C_out,
                C_out,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
                groups=groups,
                bias=use_bias,
                norm=get_norm(norm, C_out)
        )
        weight_init.c2_xavier_fill(self.conv1)
        weight_init.c2_xavier_fill(self.conv2)


        if self.stride==1:
            self.downsample = Conv2d(
                    C_in,
                    C_out,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    dilation=dilation,
                    groups=groups,
                    bias=use_bias,
                    norm=get_norm(norm, C_out)
            )
            weight_init.c2_xavier_fill(self.downsample)
    
    def forward(self, x):
        out = F.interpolate(x, size=(int(x.size(2))//2, int(x.size(3))//2), mode='bilinear', align_corners=self.align_corners)
        out = self.conv1(out)
        out = self.conv2(out)
        if self.stride == 1:
            out = F.interpolate(out, size=(int(x.size(2)), int(x.size(3))), mode='bilinear', align_corners=self.align_corners)
            out = out + self.downsample(x)
        out = self.relu(out)
        return out


class FactorizedReduce(nn.Module):
    def __init__(self, norm, C_in, C_out, kernel_size=3, stride=1, dilation=1, groups=1, use_bias=False):
        super(FactorizedReduce, self).__init__()
        assert stride in [1, 2]
        assert C_out % 2 == 0
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        if stride == 1:
            self.conv1 = Conv2d(
                C_in,
                C_out,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=dilation,
                groups=groups,
                bias=use_bias,
                norm=get_norm(norm, C_out),
                activation=F.relu,
            )
            weight_init.c2_xavier_fill(self.conv1)
            
        elif stride == 2:
            self.conv1 = Conv2d(
                C_in,
                C_out // 2,
                kernel_size=1,
                stride=2,
                padding=0,
                dilation=dilation,
                groups=groups,
                bias=use_bias,
                norm=get_norm(norm, C_out // 2),
                activation=F.relu,
            )

            self.conv2 = Conv2d(
                C_in,
                C_out // 2,
                kernel_size=1,
                stride=2,
                padding=0,
                dilation=dilation,
                groups=groups,
                bias=use_bias,
                norm=get_norm(norm, C_out // 2),
                activation=F.relu,
            )
            weight_init.c2_xavier_fill(self.conv1)
            weight_init.c2_xavier_fill(self.conv2)

    def forward(self, x):
        if self.stride == 2:
            out = torch.cat([self.conv1(x), self.conv2(x[:,:,1:,1:])], dim=1)
        else:
            out = self.conv1(x)
        return out

def pair(x):
    return (x, x) if not isinstance(x, tuple) else x

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def rel_to_abs(x):
    b, h, l, _, device, dtype = *x.shape, x.device, x.dtype
    dd = {'device': device, 'dtype': dtype}
    col_pad = torch.zeros((b, h, l, 1), **dd)
    x = torch.cat((x, col_pad), dim = 3)
    flat_x = rearrange(x, 'b h l c -> b h (l c)')
    flat_pad = torch.zeros((b, h, l - 1), **dd)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 2)
    final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
    final_x = final_x[:, :, :l, (l-1):]
    return final_x

def relative_logits_1d(q, rel_k):
    b, heads, h, w, dim = q.shape
    logits = einsum('b h x y d, r d -> b h x y r', q, rel_k)
    logits = rearrange(logits, 'b h x y r -> b (h x) y r')
    logits = rel_to_abs(logits)
    logits = logits.reshape(b, heads, h, w, w)
    logits = expand_dim(logits, dim = 3, k = h)
    return logits

# positional embeddings

class AbsPosEmb(nn.Module):
    def __init__(
        self,
        fmap_size,
        dim_head
    ):
        super().__init__()
        height, width = pair(fmap_size)
        scale = dim_head ** -0.5
        self.height = nn.Parameter(torch.randn(height, dim_head) * scale)
        self.width = nn.Parameter(torch.randn(width, dim_head) * scale)

    def forward(self, q):
        emb = rearrange(self.height, 'h d -> h () d') + rearrange(self.width, 'w d -> () w d')
        emb = rearrange(emb, ' h w d -> (h w) d')
        logits = einsum('b h i d, j d -> b h i j', q, emb)
        return logits

class RelPosEmb(nn.Module):
    def __init__(
        self,
        fmap_size,
        dim_head
    ):
        super().__init__()
        height, width = pair(fmap_size)
        scale = dim_head ** -0.5
        self.fmap_size = fmap_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        h, w = self.fmap_size

        q = rearrange(q, 'b h (x y) d -> b h x y d', x = h, y = w)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b h x i y j-> b h (x y) (i j)')

        q = rearrange(q, 'b h x y d -> b h y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b h x i y j -> b h (y x) (j i)')
        return rel_logits_w + rel_logits_h

# classes

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        heads = 4,
        dim_head = 128,
        rel_pos_emb = False
    ):
        super().__init__()
        self.heads = heads

        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)

        rel_pos_class = AbsPosEmb if not rel_pos_emb else RelPosEmb
        self.pos_emb = rel_pos_class(fmap_size, dim_head)

    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        q, k, v = self.to_qkv(fmap).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), (q, k, v))

        q *= self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim += self.pos_emb(q)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return out

class Self_Attn(nn.Module):
    def __init__(
        self,
        norm,
        dim,
        fmap_size,
        dim_out,
        downsample,
        rel_pos_emb = False
    ):
        super().__init__()

        # shortcut
        if dim != dim_out or downsample:
            self.sk = False
            kernel_size, stride, padding = (3, 2, 1) if downsample else (1, 1, 0)
            self.shortcut = Conv2d(
                dim,
                dim_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
                norm=get_norm(norm, dim_out),
                activation=F.relu,
            )
        else:
            self.sk = True
            self.shortcut = nn.Identity()

        # attn_dim_in = dim_out // proj_factor
        attn_dim_in = dim_out
        # attn_dim_out = heads * dim_head
        attn_dim_out = attn_dim_in
        activation = F.relu

        self.net1 = Conv2d(
                dim,
                attn_dim_in,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
                norm=get_norm(norm, attn_dim_in),
                activation=F.relu,
        )

        self.net2 = nn.Sequential(
            ATT(attn_dim_in),
            nn.AvgPool2d((2, 2)) if downsample else nn.Identity(),
            get_norm(norm, attn_dim_in),
            nn.ReLU(),
        )

        self.net3 = Conv2d(
                attn_dim_out,
                dim_out,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
                norm=get_norm(norm, dim_out)
        )
        
        weight_init.c2_xavier_fill(self.net1)
        weight_init.c2_xavier_fill(self.net3)
        # init last batch norm gamma to zero

        nn.init.zeros_(self.net3.norm.weight)

        # final activation

        self.activation = activation
    
    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.net1(x)
        x = self.net2(x)
        x = self.net3(x)
        x += shortcut
        return self.activation(x)

class SAHead(nn.Module):
    def __init__(
        self,
        norm,
        dim,
        fmap_size,
        dim_out,
        downsample,
        rel_pos_emb = False,
        align_corners = False
    ):
        super().__init__()

        self.stride = 2 if downsample else 1
        # shortcut
        self.align_corners = align_corners
        if dim != dim_out or downsample:
            self.sk = False
            kernel_size, stride, padding = (3, 2, 1) if downsample else (1, 1, 0)
            self.shortcut = Conv2d(
                dim,
                dim_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
                norm=get_norm(norm, dim_out),
                activation=F.relu,
            )
        else:
            self.sk = True
            self.shortcut = nn.Identity()

        # attn_dim_in = dim_out // proj_factor
        attn_dim_in = dim_out
        # attn_dim_out = heads * dim_head
        attn_dim_out = attn_dim_in
        activation = F.relu

        self.net1 = Conv2d(
                dim,
                attn_dim_in,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
                norm=get_norm(norm, attn_dim_in),
                activation=F.relu,
        )

        self.net2 = nn.Sequential(
            ATT(attn_dim_in),
            nn.AvgPool2d((2, 2)) if downsample else nn.Identity(),
            get_norm(norm, attn_dim_in),
            nn.ReLU(),
        )

        self.net3 = Conv2d(
                attn_dim_out,
                dim_out,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
                norm=get_norm(norm, dim_out)
        )
        
        weight_init.c2_xavier_fill(self.net1)
        weight_init.c2_xavier_fill(self.net3)
        # init last batch norm gamma to zero

        nn.init.zeros_(self.net3.norm.weight)

        # final activation

        self.activation = activation
    
    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.net1(x)
        x = F.interpolate(x, size=(int(x.size(2))//2, int(x.size(3))//2), mode='bilinear', align_corners=self.align_corners)
        x = self.net2(x)
        if self.stride == 1:
            x = F.interpolate(x, size=(int(x.size(2)*2), int(x.size(3)*2)), mode='bilinear', align_corners=self.align_corners)
        
        x = self.net3(x)
        x += shortcut
        return self.activation(x)

class ATT(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(ATT, self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize, -1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1, width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query, proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value, attention.permute(0,2,1) )
        out = out.view(m_batchsize, C, width,height)
        
        out = self.gamma*out + x
        return out

from collections import OrderedDict
OPS = {
    'skip' : lambda norm, C_in, C_out, stride, use_bias, fmap_size: FactorizedReduce(norm, C_in, C_out, stride, use_bias=use_bias),
    'conv' : lambda norm, C_in, C_out, stride, use_bias, fmap_size: BasicResidual1x(norm, C_in, C_out, kernel_size=3, stride=stride, dilation=1, use_bias=use_bias),
    'conv_downup' : lambda norm, C_in, C_out, stride, use_bias, fmap_size: BasicResidual_downup_1x(norm, C_in, C_out, kernel_size=3, stride=stride, dilation=1, use_bias=use_bias),
    'conv_2x' : lambda norm, C_in, C_out, stride, use_bias, fmap_size: BasicResidual2x(norm, C_in, C_out, kernel_size=3, stride=stride, dilation=1, use_bias=use_bias),
    'conv_2x_downup' : lambda norm, C_in, C_out, stride, use_bias, fmap_size: BasicResidual_downup_2x(norm, C_in, C_out, kernel_size=3, stride=stride, dilation=1, use_bias=use_bias),
    'sa': lambda norm, C_in, C_out, stride, use_bias, fmap_size: Self_Attn(norm, dim=C_in, fmap_size=(128, 256), dim_out=C_out, downsample=(stride==2))
}

OPS_name = ["FactorizedReduce", "BasicResidual1x", "BasicResidual_downup_1x", "BasicResidual2x", "BasicResidual_downup_2x", "Self_Attn"]

OPS_Class = OrderedDict()
OPS_Class['skip'] = FactorizedReduce
OPS_Class['conv'] = BasicResidual1x
OPS_Class['conv_downup'] = BasicResidual_downup_1x
OPS_Class['conv_2x'] = BasicResidual2x
OPS_Class['conv_2x_downup'] = BasicResidual_downup_2x
OPS_Class['sa'] = Self_Attn
