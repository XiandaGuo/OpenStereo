# @Time    : 2023/10/8 05:02
# @Author  : zhangchenming
import torch
import torch.nn as nn
from stereo.modeling.common.basic_block_3d import BasicConv3d
from stereo.modeling.common.basic_block_2d import BasicConv2d


class CoExCostVolume(nn.Module):
    def __init__(self, maxdisp, group=1):
        super(CoExCostVolume, self).__init__()
        self.maxdisp = maxdisp + 1
        self.group = group
        self.unfold = nn.Unfold((1, maxdisp + 1), 1, 0, 1)
        self.left_pad = nn.ZeroPad2d((maxdisp, 0, 0, 0))

    def forward(self, x, y):
        b, c, h, w = x.shape

        y = self.left_pad(y)
        unfolded_y = self.unfold(y)
        unfolded_y = unfolded_y.reshape(b, self.group, c // self.group, self.maxdisp, h, w)

        x = x.reshape(b, self.group, c // self.group, 1, h, w)

        cost = (x * unfolded_y).sum(2)
        cost = torch.flip(cost, dims=[2])

        return cost


def correlation_volume(left_feature, right_feature, max_disp):
    b, c, h, w = left_feature.size()
    cost_volume = left_feature.new_zeros(b, max_disp, h, w)
    for i in range(max_disp):
        if i > 0:
            cost_volume[:, i, :, i:] = (left_feature[:, :, :, i:] * right_feature[:, :, :, :-i]).mean(dim=1)
        else:
            cost_volume[:, i, :, :] = (left_feature * right_feature).mean(dim=1)
    cost_volume = cost_volume.contiguous()
    return cost_volume


def compute_volume(reference_embedding, target_embedding, maxdisp, side='left'):
    batch, channel, height, width = reference_embedding.size()

    cost = torch.zeros(batch, channel, maxdisp, height, width, device='cuda').type_as(reference_embedding)
    cost[:, :, 0, :, :] = reference_embedding - target_embedding
    for idx in range(1, maxdisp):
        if side == 'left':
            cost[:, :, idx, :, idx:] = reference_embedding[:, :, :, idx:] - target_embedding[:, :, :, :-idx]
        if side == 'right':
            cost[:, :, idx, :, :-idx] = target_embedding[:, :, :, idx:] - reference_embedding[:, :, :, :-idx]
    cost = cost.contiguous()

    return cost


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W], requires_grad=False)
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def build_corr_volume(img_left, img_right, max_disp):
    B, C, H, W = img_left.shape
    volume = img_left.new_zeros([B, max_disp, H, W])
    for i in range(max_disp):
        if (i > 0) & (i < W):
            volume[:, i, :, i:] = (img_left[:, :, :, i:] * img_right[:, :, :, :W-i]).mean(dim=1)
        else:
            volume[:, i, :, :] = (img_left[:, :, :, :] * img_right[:, :, :, :]).mean(dim=1)

    volume = volume.contiguous()
    return volume


def build_sub_volume(feat_l, feat_r, maxdisp):
    cost = torch.zeros((feat_l.size()[0], maxdisp, feat_l.size()[2], feat_l.size()[3]), device='cuda')
    for i in range(maxdisp):
        cost[:, i, :, :i] = feat_l[:, :, :, :i].abs().sum(1)
        if i > 0:
            cost[:, i, :, i:] = torch.norm(feat_l[:, :, :, i:] - feat_r[:, :, :, :-i], 1, 1)
        else:
            cost[:, i, :, i:] = torch.norm(feat_l[:, :, :, :] - feat_r[:, :, :, :], 1, 1)

    return cost.contiguous()


class InterlacedVolume(nn.Module):
    def __init__(self, num_features=8):
        super(InterlacedVolume, self).__init__()
        self.num_features = num_features

        self.conv3d = nn.Sequential(BasicConv3d(in_channels=1, out_channels=16,
                                                norm_layer=nn.BatchNorm3d, act_layer=nn.ReLU,
                                                kernel_size=(8, 3, 3), stride=(8, 1, 1), padding=(0, 1, 1)),
                                    BasicConv3d(in_channels=16, out_channels=32,
                                                norm_layer=nn.BatchNorm3d, act_layer=nn.ReLU,
                                                kernel_size=(8, 3, 3), stride=(8, 1, 1), padding=(0, 1, 1)),
                                    BasicConv3d(in_channels=32, out_channels=16,
                                                norm_layer=nn.BatchNorm3d, act_layer=nn.ReLU,
                                                kernel_size=(3, 3, 3), stride=(3, 1, 1), padding=(0, 1, 1))
                                    )

        self.volume11 = BasicConv2d(in_channels=16, out_channels=self.num_features,
                                    norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU,
                                    kernel_size=1, stride=1)

    @staticmethod
    def interweave_tensors(refimg_fea, targetimg_fea):
        B, C, H, W = refimg_fea.shape
        interwoven_features = refimg_fea.new_zeros([B, 2 * C, H, W])  # [bz, 192, H/4, W/4]
        interwoven_features[:, ::2, :, :] = refimg_fea
        interwoven_features[:, 1::2, :, :] = targetimg_fea
        interwoven_features = interwoven_features.contiguous()
        return interwoven_features

    def forward(self, feat_l, feat_r, maxdisp):
        B, C, H, W = feat_l.shape
        volume = feat_l.new_zeros([B, self.num_features, maxdisp, H, W])
        for i in range(maxdisp):
            if i > 0:
                x = self.interweave_tensors(feat_l[:, :, :, i:], feat_r[:, :, :, :-i])
                x = torch.unsqueeze(x, 1)
                x = self.conv3d(x)
                x = torch.squeeze(x, 2)
                x = self.volume11(x)
                volume[:, :, i, :, i:] = x
            else:
                x = self.interweave_tensors(feat_l, feat_r)  # [bz, 192, H/4, W/4]
                x = torch.unsqueeze(x, 1)  # [bz, 1, 192, H/4, W/4]
                x = self.conv3d(x)  # [bz, 16, 1, H/4, W/4]
                x = torch.squeeze(x, 2)  # [bz, 16, H/4, W/4]
                x = self.volume11(x)  # [bz, self.num_features, H/4, W/4]
                volume[:, :, i, :, :] = x

        volume = volume.contiguous()
        return volume
