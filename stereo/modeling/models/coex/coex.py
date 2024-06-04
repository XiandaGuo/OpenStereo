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

        if self.training:
            return {'disp_preds': disp_out['disp_ests'],
                    'disp_pred': disp_out['disp_ests'][0]}
        else:
            return {'disp_pred': disp_out['inference_disp']['disp_est']}

    def get_loss(self, model_preds, input_data):
        disp_gt = input_data["disp"]  # [bz, h, w]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, h, w]

        weights = [1.0, 0.3]

        loss = 0.0
        for disp_est, weight in zip(model_preds['disp_preds'], weights):
            loss += weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True)

        loss = loss * 0.77
        loss_info = {'scalar/train/loss_disp': loss.item()}
        return loss, loss_info
