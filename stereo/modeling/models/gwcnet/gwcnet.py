# @Time    : 2024/4/2 12:31
# @Author  : zhangchenming
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gwcnet_backbone import GwcNet as GwcNetBackbone
from .gwcnet_cost_processor import GwcVolumeCostProcessor
from .gwcnet_disp_processor import GwcDispProcessor


class GwcNet(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.maxdisp = cfgs.MAX_DISP

        use_concat_volume = cfgs.USE_CONCAT_VOLUME
        concat_channels = cfgs.CONCAT_CHANNELS
        downsample = cfgs.DOWNSAMPLE
        num_groups = cfgs.NUM_GROUPS

        self.Backbone = GwcNetBackbone(use_concat_volume=use_concat_volume, concat_channels=concat_channels)
        self.CostProcessor = GwcVolumeCostProcessor(maxdisp=self.maxdisp, downsample=downsample, num_groups=num_groups,
                                                    use_concat_volume=use_concat_volume)
        self.DispProcessor = GwcDispProcessor(maxdisp=self.maxdisp, downsample=downsample, num_groups=num_groups,
                                              use_concat_volume=use_concat_volume, concat_channels=concat_channels)

    def forward(self, inputs):
        """Forward the network."""
        backbone_out = self.Backbone(inputs)
        inputs.update(backbone_out)
        cost_out = self.CostProcessor(inputs)
        inputs.update(cost_out)
        disp_out = self.DispProcessor(inputs)

        if self.training:
            return {'disp_preds': disp_out['training_disp']['disp']['disp_ests'],
                    'disp_pred': disp_out['training_disp']['disp']['disp_ests'][-1]}

        return {'disp_pred': disp_out['inference_disp']['disp_est']}


    def get_loss(self, model_preds, input_data):
        disp_gt = input_data["disp"]  # [bz, h, w]
        mask = (disp_gt < self.maxdisp) & (disp_gt > 0)  # [bz, h, w]

        weights = [0.5, 0.5, 0.7, 1.0]

        loss = 0.0
        for disp_est, weight in zip(model_preds['disp_preds'], weights):
            loss += weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True)

        loss_info = {'scalar/train/loss_disp': loss.item()}
        return loss, loss_info
