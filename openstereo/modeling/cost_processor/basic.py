import torch
import torch.nn as nn
from torch.autograd import Variable


class BasicCostVolumeProcessor(nn.Module):
    def __init__(self, maxdisp, downsample, *args, **kwargs):
        super().__init__()
        self.maxdisp = maxdisp
        self.downsample = downsample

    def forward(self, inputs):
        left_feature = inputs["left_feature"]
        right_feature = inputs["right_feature"]
        B, C, H, W = left_feature.shape
        cost_volume = Variable(torch.zeros(B, C * 2, self.maxdisp // self.downsample, H, W, device=left_feature.device))
        for i in range(self.maxdisp // self.downsample):
            if i > 0:
                cost_volume[:, :C, i, :, i:] = left_feature[:, :, :, i:]
                cost_volume[:, C:, i, :, i:] = right_feature[:, :, :, :-i]
            else:
                cost_volume[:, :C, i, :, :] = left_feature
                cost_volume[:, C:, i, :, :] = right_feature
        return cost_volume.contiguous()
