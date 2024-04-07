import torch
import torch.nn as nn
from stereo.modeling.common.basic_block_3d import BasicConv3d, BasicDeconv3d
from stereo.modeling.models.igev.igev_blocks import FeatureAtt


class Hourglass(nn.Module):
    def __init__(self, in_channels, backbone_channels=None):
        super(Hourglass, self).__init__()

        if backbone_channels is None:
            backbone_channels = [48, 64, 192, 120]

        self.conv1 = nn.Sequential(
            BasicConv3d(in_channels, in_channels * 2,
                        norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                        kernel_size=3, padding=1, stride=2, dilation=1),
            BasicConv3d(in_channels * 2, in_channels * 2,
                        norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                        kernel_size=3, padding=1, stride=1, dilation=1),
        )

        self.conv2 = nn.Sequential(
            BasicConv3d(in_channels * 2, in_channels * 4,
                        norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                        kernel_size=3, padding=1, stride=2, dilation=1),
            BasicConv3d(in_channels * 4, in_channels * 4,
                        norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                        kernel_size=3, padding=1, stride=1, dilation=1))

        self.conv3 = nn.Sequential(
            BasicConv3d(in_channels * 4, in_channels * 6,
                        norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                        kernel_size=3, padding=1, stride=2, dilation=1),
            BasicConv3d(in_channels * 6, in_channels * 6,
                        norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                        kernel_size=3, padding=1, stride=1, dilation=1))

        self.conv3_up = BasicDeconv3d(in_channels * 6, in_channels * 4,
                                      norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                                      kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicDeconv3d(in_channels * 4, in_channels * 2,
                                      norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                                      kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicDeconv3d(in_channels * 2, in_channels,
                                      norm_layer=None, act_layer=None,
                                      kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(
            BasicConv3d(in_channels * 8, in_channels * 4,
                        norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                        kernel_size=1, padding=0, stride=1),
            BasicConv3d(in_channels * 4, in_channels * 4,
                        norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                        kernel_size=3, padding=1, stride=1),
            BasicConv3d(in_channels * 4, in_channels * 4,
                        norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                        kernel_size=3, padding=1, stride=1), )

        self.agg_1 = nn.Sequential(
            BasicConv3d(in_channels * 4, in_channels * 2,
                        norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                        kernel_size=1, padding=0, stride=1),
            BasicConv3d(in_channels * 2, in_channels * 2,
                        norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                        kernel_size=3, padding=1, stride=1),
            BasicConv3d(in_channels * 2, in_channels * 2,
                        norm_layer=nn.BatchNorm3d, act_layer=nn.LeakyReLU,
                        kernel_size=3, padding=1, stride=1))

        self.feature_att_8 = FeatureAtt(in_channels * 2, backbone_channels[1])
        self.feature_att_16 = FeatureAtt(in_channels * 4, backbone_channels[2])
        self.feature_att_32 = FeatureAtt(in_channels * 6, backbone_channels[3])
        self.feature_att_up_16 = FeatureAtt(in_channels * 4, backbone_channels[2])
        self.feature_att_up_8 = FeatureAtt(in_channels * 2, backbone_channels[1])

    def forward(self, x, features, return_multi=False):  # [bz, c, disp/4, H/4, W/4]
        conv1 = self.conv1(x)  # [bz, 2c, disp/8, H/8, W/8]
        conv1 = self.feature_att_8(conv1, features[1])  # [bz, 2c, disp/8, H/8, W/8]

        conv2 = self.conv2(conv1)  # [bz, 4c, disp/16, H/16, W/16]
        conv2 = self.feature_att_16(conv2, features[2])  # [bz, 4c, disp/16, H/16, W/16]

        conv3 = self.conv3(conv2)  # [bz, 6c, disp/32, H/32, W/32]
        conv3 = self.feature_att_32(conv3, features[3])  # [bz, 6c, disp/32, H/32, W/32]

        conv3_up = self.conv3_up(conv3)  # [bz, 4c, disp/16, H/16, W/16]
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)  # [bz, 4c, disp/16, H/16, W/16]
        conv2 = self.feature_att_up_16(conv2, features[2])  # [bz, 4c, disp/16, H/16, W/16]

        conv2_up = self.conv2_up(conv2)  # [bz, 2c, disp/8, H/8, W/8]
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])  # [bz, 2c, disp/8, H/8, W/8]

        conv = self.conv1_up(conv1)  # [bz, c, disp/4, H/4, W/4]

        if return_multi:
            return [conv, conv1, conv2]
        else:
            return conv
