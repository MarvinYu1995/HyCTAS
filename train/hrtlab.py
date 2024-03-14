import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from pdb import set_trace as bp
from utils.pyt_utils import load_pretrain
from collections import OrderedDict

import hrtnet
from hrtnet import HyCTAS, HRTSemHeadLight, HRTInsHeadLight

class HRTLab(nn.Module):
    def __init__(self, ops, paths, downs, widths, lasts, semantic_loss, semantic_loss_weight, center_loss, center_loss_weight, offset_loss, offset_loss_weight, lamb=0.2, eval_flag=True, num_classes=19, layers=9, Fch=12, width_mult_list=[1.,], stem_head_width=(1., 1.), input_size=(1024, 2048), norm='naiveSyncBN', align_corners=False, pretrain='', model_type='0', sem_only=True, use_aux=True):
        """
        model_type: 0 s=8, no fusion, 1, fuse s4, 2 fuse s4, s8
        """
        super(HRTLab, self).__init__()
        self._num_classes = num_classes
        assert layers >= 2
        self._layers = layers
        self._Fch = Fch
        self._width_mult_list = width_mult_list
        self._stem_head_width = stem_head_width
        self.align_corners = align_corners
        self.input_size = input_size
        self.lasts = lasts
        self.use_aux = use_aux
        self.sem_only = sem_only
        self.model_type = model_type

        self.ops = ops
        self.paths = paths
        self.downs = downs
        self.widths = widths
        self._branch = len(lasts)
        self.lasts = lasts
        self.shape8 = [None, None]
        self.shape16 = [None, None]

        self.backbone = HyCTAS(ops, paths, downs, widths, lasts, norm=norm, layers=layers, Fch=Fch, width_mult_list=[4./12, 6./12, 8./12, 10./12, 1.,], stem_head_width=stem_head_width, align_corners=align_corners, input_size=input_size)
        
        if pretrain:
            self.backbone = load_pretrain(self.backbone, pretrain)
        
        self.ins_head = None
        self.ins_head_aux16 = None
        self.ins_head_aux32 = None
        self.sem_head = None
        self.sem_head_aux16 = None
        self.sem_head_aux32 = None

        if model_type==0:
            ins_head = getattr(hrtnet, 'HRTSemHeadLight')
            sem_head = getattr(hrtnet, 'HRTSemHeadLight')
        elif model_type==1:
            ins_head = getattr(hrtnet, 'HRTInsHead1')
            sem_head = getattr(hrtnet, 'HRTSemHead1')
        elif model_type==2:
            ins_head = getattr(hrtnet, 'HRTInsHead2')
            sem_head = getattr(hrtnet, 'HRTSemHead2')
        else:
            raise NotImplementedError

        self.sem_head = sem_head(norm, self.backbone.outs8_inplanes, self.backbone.s4_inplanes, self.backbone.s8_inplanes, out_planes=num_classes, fmap_size=input_size, align_corners=self.align_corners)
        if self.use_aux:
            if self.backbone.outs32_inplanes is not None:
                self.sem_head_aux32 =HRTSemHeadLight(norm, self.backbone.outs32_inplanes, self.backbone.s4_inplanes, self.backbone.s8_inplanes, out_planes=num_classes, fmap_size=input_size, align_corners=self.align_corners)
            if self.backbone.outs16_inplanes is not None:
                self.sem_head_aux16 =HRTSemHeadLight(norm, self.backbone.outs16_inplanes, self.backbone.s4_inplanes, self.backbone.s8_inplanes, out_planes=num_classes, fmap_size=input_size, align_corners=self.align_corners)
        if not sem_only:
            self.ins_head = ins_head(norm, self.backbone.outs8_inplanes, self.backbone.s4_inplanes, self.backbone.s8_inplanes, fmap_size=input_size, align_corners=self.align_corners)
            if self.use_aux:
                if self.backbone.outs32_inplanes is not None:
                    self.ins_head_aux32 =HRTInsHeadLight(norm, self.backbone.outs32_inplanes, self.backbone.s4_inplanes, self.backbone.s8_inplanes, fmap_size=input_size, align_corners=self.align_corners)
                if self.backbone.outs16_inplanes is not None:
                    self.ins_head_aux16 =HRTInsHeadLight(norm, self.backbone.outs16_inplanes, self.backbone.s4_inplanes, self.backbone.s8_inplanes, fmap_size=input_size, align_corners=self.align_corners)

        self.eval_flag = eval_flag
        self.lamb = lamb
        self.semantic_loss = semantic_loss
        self.semantic_loss_weight = semantic_loss_weight
        self.center_loss = center_loss
        self.center_loss_weight = center_loss_weight
        self.offset_loss = offset_loss
        self.offset_loss_weight = offset_loss_weight

    def _upsample_predictions(self, pred, input_shape):
        """Upsamples final prediction, with special handling to offset.
            Args:
                pred (dict): stores all output of the segmentation model.
                input_shape (tuple): spatial resolution of the desired shape.
            Returns:
                result (OrderedDict): upsampled dictionary.
            """
        # Override upsample method to correctly handle `offset`
        result = OrderedDict()
        for key in pred.keys():
            out = F.interpolate(pred[key], size=input_shape, mode='bilinear', align_corners=self.align_corners)
            if 'offset' in key:
                scale = (input_shape[0] - 1) // (pred[key].shape[2] - 1)
                out *= scale
            result[key] = out
        return result

    def forward(self, input, targets=None):
        input_shape = input.shape[-2:]
        H, W = input_shape
        results8 = {}
        results16 = {}
        results32 = {}

        low_level_feature_s4, low_level_feature_s8, s8_feat, s16_feat, s32_feat = self.backbone(input)
        if self.model_type == 0:
            results8['semantic'] = self.sem_head(s8_feat)
        else:
            results8['semantic'] = self.sem_head([low_level_feature_s4, low_level_feature_s8, s8_feat])
        
        if not self.sem_only:
            if self.model_type == 0:
                results8['center'],  results8['offset'] = self.ins_head(s8_feat)
            else:
                results8['center'],  results8['offset'] = self.ins_head([low_level_feature_s4, low_level_feature_s8, s8_feat])
        
        results8 = self._upsample_predictions(results8, input_shape)
        if self.training:
            loss = {}
            if self.use_aux:
                if s16_feat is not None:
                    results16['semantic'] = self.sem_head_aux16(s16_feat)
                    if not self.sem_only:
                        results16['center'],  results16['offset'] = self.ins_head_aux16(s16_feat)
                else:
                    results16 = None
                if s32_feat is not None:
                    results32['semantic'] = self.sem_head_aux32(s32_feat)
                    if not self.sem_only:
                        results32['center'],  results32['offset'] = self.ins_head_aux32(s32_feat)
                else:
                    results32 = None
            
            if results16 is not None: results16 = self._upsample_predictions(results16, input_shape)
            if results32 is not None: results32 = self._upsample_predictions(results32, input_shape)

            if 'semantic_weights' in targets.keys():
                semantic_loss = self.semantic_loss(
                    results8['semantic'], targets['semantic'], semantic_weights=targets['semantic_weights'])
                if self.use_aux:
                    if results16 is not None:
                        semantic_loss = semantic_loss + self.lamb * self.semantic_loss(
                        results16['semantic'], targets['semantic'], semantic_weights=targets['semantic_weights']) 
                    if results32 is not None:
                        semantic_loss = semantic_loss + self.lamb * self.semantic_loss(
                        results32['semantic'], targets['semantic'], semantic_weights=targets['semantic_weights'])
            else:
                semantic_loss = self.semantic_loss(results8['semantic'], targets['semantic'])
                if self.use_aux:
                    if results16 is not None:
                        semantic_loss = semantic_loss + self.lamb * self.semantic_loss(results16['semantic'], targets['semantic']) 
                    if results32 is not None:
                        semantic_loss = semantic_loss + self.lamb * self.semantic_loss(results32['semantic'], targets['semantic'])

            if self.sem_only:
                total_loss = semantic_loss
                loss['semantic'] = semantic_loss
                loss['total'] = total_loss
            else:
                center_loss_weights = targets['center_weights'][:, None, :, :].expand_as(results8['center'])
                center_loss = self.center_loss(results8['center'], targets['center'])
                if results16 is not None:
                    center_loss = center_loss + self.lamb * self.center_loss(results16['center'], targets['center'])
                if results32 is not None:
                    center_loss = center_loss + self.lamb * self.center_loss(results32['center'], targets['center'])
                center_loss = center_loss * center_loss_weights
                # safe division
                if center_loss_weights.sum() > 0:
                    center_loss = center_loss.sum() / center_loss_weights.sum() * self.center_loss_weight
                else:
                    center_loss = center_loss.sum() * 0

                # Pixel-wise loss weight
                offset_loss_weights = targets['offset_weights'][:, None, :, :].expand_as(results8['offset'])
                offset_loss = self.offset_loss(results8['offset'], targets['offset'])
                if results16 is not None:
                    offset_loss = offset_loss + self.lamb * self.offset_loss(results16['offset'], targets['offset']) 
                if results32 is not None:
                    offset_loss = offset_loss + self.lamb * self.offset_loss(results32['offset'], targets['offset']) 
                offset_loss =  offset_loss * offset_loss_weights
                # safe division
                if offset_loss_weights.sum() > 0:
                    offset_loss = offset_loss.sum() / offset_loss_weights.sum() * self.offset_loss_weight
                else:
                    offset_loss = offset_loss.sum() * 0

                semantic_loss = semantic_loss * self.semantic_loss_weight
                total_loss = semantic_loss + center_loss + offset_loss
                loss['semantic'] = semantic_loss
                loss['center'] = center_loss
                loss['offset'] = offset_loss
                loss['total'] = total_loss
            return loss
        else:
            if self.sem_only:
                return results8['semantic']
            else:
                if self.eval_flag:
                    return results8['semantic']
                else:
                    return results8

    def freeze_semantic(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.sem_head.parameters():
            p.requires_grad = False
        if self.sem_head_aux16 is not None:
            for p in self.sem_head_aux16.parameters():
                p.requires_grad = False
        if self.sem_head_aux32 is not None:
            for p in self.sem_head_aux32.parameters():
                p.requires_grad = False
      
    # def train(self, mode=True):
    #     self.training = mode
    #     if mode == False:
    #         super(Network_Multi_Path_Infer_Panoptic, self).train(False)
    #     else:
    #         self.train(False)
    #         self.heads8.train(mode)
    #         self.heads16.train(mode)
    #         self.heads32.train(mode)
    #         self.training = True

    #     return self

            

