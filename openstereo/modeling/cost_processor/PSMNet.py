import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from modeling.common.basic_layers import conv3d_bn_relu, conv3d_bn, deconv3d_bn


def cat_fms(reference_fm, target_fm, max_disp=192, start_disp=0, dilation=1):
    """
    Concat left and right in Channel dimension to form the raw cost volume.
    Args:
        max_disp, (int): under the scale of feature used,
            often equals to (end disp - start disp + 1), the maximum searching range of disparity
        start_disp (int): the start searching disparity index, usually be 0
            dilation (int): the step between near disparity index
        dilation (int): the step between near disparity index

    Inputs:
        reference_fm, (Tensor): reference feature, i.e. left image feature, in [BatchSize, Channel, Height, Width] layout
        target_fm, (Tensor): target feature, i.e. right image feature, in [BatchSize, Channel, Height, Width] layout

    Output:
        concat_fm, (Tensor): the formed cost volume, in [BatchSize, Channel*2, disp_sample_number, Height, Width] layout

    """
    device = reference_fm.device
    N, C, H, W = reference_fm.shape

    end_disp = start_disp + max_disp - 1
    disp_sample_number = (max_disp + dilation - 1) // dilation
    disp_index = torch.linspace(start_disp, end_disp, disp_sample_number)

    concat_fm = torch.zeros(N, C * 2, disp_sample_number, H, W).to(device)
    idx = 0
    for i in disp_index:
        i = int(i)  # convert torch.Tensor to int, so that it can be index
        if i > 0:
            concat_fm[:, :C, idx, :, i:] = reference_fm[:, :, :, i:]
            concat_fm[:, C:, idx, :, i:] = target_fm[:, :, :, :-i]
        elif i == 0:
            concat_fm[:, :C, idx, :, :] = reference_fm
            concat_fm[:, C:, idx, :, :] = target_fm
        else:
            concat_fm[:, :C, idx, :, :i] = reference_fm[:, :, :, :i]
            concat_fm[:, C:, idx, :, :i] = target_fm[:, :, :, abs(i):]
        idx = idx + 1

    concat_fm = concat_fm.contiguous()
    return concat_fm


class Hourglass(nn.Module):
    """
    An implementation of hourglass module proposed in PSMNet.
    Args:
        in_planes (int): the channels of raw cost volume
        batch_norm (bool): whether use batch normalization layer,
            default True
    Inputs:
        x, (Tensor): cost volume
            in [BatchSize, in_planes, MaxDisparity, Height, Width] layout
        presqu, (optional, Tensor): cost volume
            in [BatchSize, in_planes * 2, MaxDisparity, Height/2, Width/2] layout
        postsqu, (optional, Tensor): cost volume
            in [BatchSize, in_planes * 2, MaxDisparity, Height/2, Width/2] layout
    Outputs:
        out, (Tensor): cost volume
            in [BatchSize, in_planes, MaxDisparity, Height, Width] layout
        pre, (optional, Tensor): cost volume
            in [BatchSize, in_planes * 2, MaxDisparity, Height/2, Width/2] layout
        post, (optional, Tensor): cost volume
            in [BatchSize, in_planes * 2, MaxDisparity, Height/2, Width/2] layout

    """

    def __init__(self, in_planes, batch_norm=True):
        super(Hourglass, self).__init__()
        self.batch_norm = batch_norm

        self.conv1 = conv3d_bn_relu(
            self.batch_norm, in_planes, in_planes * 2,
            kernel_size=3, stride=2, padding=1, bias=False
        )

        self.conv2 = conv3d_bn(
            self.batch_norm, in_planes * 2, in_planes * 2,
            kernel_size=3, stride=1, padding=1, bias=False
        )

        self.conv3 = conv3d_bn_relu(
            self.batch_norm, in_planes * 2, in_planes * 2,
            kernel_size=3, stride=2, padding=1, bias=False
        )
        self.conv4 = conv3d_bn_relu(
            self.batch_norm, in_planes * 2, in_planes * 2,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv5 = deconv3d_bn(
            self.batch_norm, in_planes * 2, in_planes * 2,
            kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
        )
        self.conv6 = deconv3d_bn(
            self.batch_norm, in_planes * 2, in_planes,
            kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
        )

    def forward(self, x, presqu=None, postsqu=None):
        # in: [B, C, D, H, W], out: [B, 2C, D, H/2, W/2]
        out = self.conv1(x)
        # in: [B, 2C, D, H/2, W/2], out: [B, 2C, D, H/2, W/2]
        pre = self.conv2(out)
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        # in: [B, 2C, D, H/2, W/2], out: [B, 2C, D, H/4, W/4]
        out = self.conv3(pre)
        # in: [B, 2C, D, H/4, W/4], out: [B, 2C, D, H/4, W/4]
        out = self.conv4(out)

        # in: [B, 2C, D, H/4, W/4], out: [B, 2C, D, H/2, W/2]
        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        # in: [B, 2C, D, H/2, W/2], out: [B, C, D, H, W]
        out = self.conv6(post)

        return out, pre, post


class PSMAggregator(nn.Module):
    """
    Args:
        max_disp (int): max disparity
        in_planes (int): the channels of raw cost volume
        batch_norm (bool): whether use batch normalization layer, default True

    Inputs:
        raw_cost (Tensor): concatenation-based cost volume without further processing,
            in [BatchSize, in_planes, MaxDisparity//4, Height//4, Width//4] layout
    Outputs:
        cost_volume (tuple of Tensor): cost volume
            in [BatchSize, MaxDisparity, Height, Width] layout
    """

    def __init__(self, max_disp, in_planes=64, batch_norm=True):
        super(PSMAggregator, self).__init__()
        self.max_disp = max_disp
        self.in_planes = in_planes
        self.batch_norm = batch_norm

        self.dres0 = nn.Sequential(
            conv3d_bn_relu(batch_norm, self.in_planes, 32, 3, 1, 1, bias=False),
            conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1, bias=False),
        )
        self.dres1 = nn.Sequential(
            conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1, bias=False),
            conv3d_bn(batch_norm, 32, 32, 3, 1, 1, bias=False)
        )
        self.dres2 = Hourglass(in_planes=32, batch_norm=batch_norm)
        self.dres3 = Hourglass(in_planes=32, batch_norm=batch_norm)
        self.dres4 = Hourglass(in_planes=32, batch_norm=batch_norm)

        self.classif1 = nn.Sequential(
            conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1, bias=False),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.classif2 = nn.Sequential(
            conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1, bias=False),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.classif3 = nn.Sequential(
            conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1, bias=False),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, raw_cost):
        B, C, D, H, W = raw_cost.shape
        # raw_cost: (BatchSize, Channels*2, MaxDisparity/4, Height/4, Width/4)
        cost0 = self.dres0(raw_cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre2, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        # (BatchSize, 1, max_disp, Height, Width)
        full_h, full_w = H * 4, W * 4
        align_corners = True
        cost1 = F.interpolate(
            cost1, [self.max_disp, full_h, full_w],
            mode='trilinear', align_corners=align_corners
        )
        cost2 = F.interpolate(
            cost2, [self.max_disp, full_h, full_w],
            mode='trilinear', align_corners=align_corners
        )
        cost3 = F.interpolate(
            cost3, [self.max_disp, full_h, full_w],
            mode='trilinear', align_corners=align_corners
        )

        # (BatchSize, max_disp, Height, Width)
        cost1 = torch.squeeze(cost1, 1)
        cost2 = torch.squeeze(cost2, 1)
        cost3 = torch.squeeze(cost3, 1)

        return [cost3, cost2, cost1]


class PSMCostProcessor(nn.Module):
    def __init__(self, max_disp=192, in_planes=64):
        super().__init__()
        self.cat_func = partial(
            cat_fms,
            max_disp=int(max_disp // 4),
            start_disp=0,
            dilation=1,
        )
        self.aggregator = PSMAggregator(
            max_disp=max_disp,
            in_planes=in_planes
        )

    def forward(self, inputs):
        # 1. build raw cost by concat
        left_feature = inputs['ref_feature']
        right_feature = inputs['tgt_feature']
        cat_cost = self.cat_func(left_feature, right_feature)
        # 2. aggregate cost by 3D-hourglass
        costs = self.aggregator(cat_cost)
        [cost3, cost2, cost1] = costs
        return {
            "cost1": cost1,
            "cost2": cost2,
            "cost3": cost3
        }

    def input_output(self):
        return {
            "inputs": ["ref_feature", "tgt_feature"],
            "outputs": ["cost1", "cost2", "cost3"]
        }
