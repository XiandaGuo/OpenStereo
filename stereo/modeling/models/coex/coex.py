# @Time    : 2024/4/2 10:53
# @Author  : zhangchenming
import torch
import torch.nn as nn
import torch.nn.functional as F
from .coex_backbone import CoExBackbone
from .coex_cost_processor import CoExCostProcessor
from .coex_disp_processor import CoExDispProcessor


class CoEx(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.max_disp = cfgs.MAX_DISP
        spixel_branch_channels = cfgs.SPIXEL_BRANCH_CHANNELS
        chans = cfgs.CHANS
        matching_weighted = cfgs.MATCHING_WEIGHTED
        matching_head = cfgs.MATCHING_HEAD
        gce = cfgs.GCE
        aggregation_disp_strides = cfgs.AGGREGATION_DISP_STRIDES
        aggregation_channels = cfgs.AGGREGATION_CHANNELS
        aggregation_blocks_num = cfgs.AGGREGATION_BLOCKS_NUM
        regression_topk = cfgs.REGRESSION_TOPK

        self.Backbone = CoExBackbone(spixel_branch_channels=spixel_branch_channels)
        self.CostProcessor = CoExCostProcessor(max_disp=self.max_disp,
                                               gce=gce,
                                               matching_weighted=matching_weighted,
                                               spixel_branch_channels=spixel_branch_channels,
                                               matching_head=matching_head,
                                               aggregation_disp_strides=aggregation_disp_strides,
                                               aggregation_channels=aggregation_channels,
                                               aggregation_blocks_num=aggregation_blocks_num,
                                               chans=chans)
        self.DispProcessor = CoExDispProcessor(max_disp=self.max_disp, regression_topk=regression_topk, chans=chans)

    def forward(self, inputs):
        """Forward the network."""
        backbone_out = self.Backbone(inputs)
        inputs.update(backbone_out)
        cost_out = self.CostProcessor(inputs)
        inputs.update(cost_out)
        disp_out = self.DispProcessor(inputs)

        return {'disp_pred': disp_out['inference_disp']['disp_est']}
