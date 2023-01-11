from __future__ import print_function
import math
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

from modeling.base_model import BaseModel

__all__ = ['GwcNet']


def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.BatchNorm3d(out_channels))


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)


def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class feature_extraction(nn.Module):
    def __init__(self, concat_feature=False, concat_feature_channel=12):
        super(feature_extraction, self).__init__()
        self.concat_feature = concat_feature

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        if self.concat_feature:
            self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        gwc_feature = torch.cat((l2, l3, l4), dim=1)

        if not self.concat_feature:
            return {"gwc_feature": gwc_feature}
        else:
            concat_feature = self.lastconv(gwc_feature)
            return {"gwc_feature": gwc_feature, "concat_feature": concat_feature}


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6


# class GNet(nn.Module):
#     def __init__(self, maxdisp, use_concat_volume=False):
#         super().__init__()
#         self.maxdisp = maxdisp
#         self.use_concat_volume = use_concat_volume
#
#         self.num_groups = 40
#
#         if self.use_concat_volume:
#             self.concat_channels = 12
#             self.feature_extraction = feature_extraction(concat_feature=True,
#                                                          concat_feature_channel=self.concat_channels)
#         else:
#             self.concat_channels = 0
#             self.feature_extraction = feature_extraction(concat_feature=False)
#
#         self.dres0 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1),
#                                    nn.ReLU(inplace=True),
#                                    convbn_3d(32, 32, 3, 1, 1),
#                                    nn.ReLU(inplace=True))
#
#         self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
#                                    nn.ReLU(inplace=True),
#                                    convbn_3d(32, 32, 3, 1, 1))
#
#         self.dres2 = hourglass(32)
#
#         self.dres3 = hourglass(32)
#
#         self.dres4 = hourglass(32)
#
#         self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
#                                       nn.ReLU(inplace=True),
#                                       nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
#
#         self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
#                                       nn.ReLU(inplace=True),
#                                       nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
#
#         self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
#                                       nn.ReLU(inplace=True),
#                                       nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
#
#         self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
#                                       nn.ReLU(inplace=True),
#                                       nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.Conv3d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm3d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.bias.data.zero_()
#
#     def forward(self, left, right):
#         features_left = self.feature_extraction(left)
#         features_right = self.feature_extraction(right)
#
#         gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4,
#                                       self.num_groups)
#         if self.use_concat_volume:
#             concat_volume = build_concat_volume(features_left["concat_feature"], features_right["concat_feature"],
#                                                 self.maxdisp // 4)
#             volume = torch.cat((gwc_volume, concat_volume), 1)
#         else:
#             volume = gwc_volume
#
#         cost0 = self.dres0(volume)
#         cost0 = self.dres1(cost0) + cost0
#
#         out1 = self.dres2(cost0)
#         out2 = self.dres3(out1)
#         out3 = self.dres4(out2)
#
#         if self.training:
#             cost0 = self.classif0(cost0)
#             cost1 = self.classif1(out1)
#             cost2 = self.classif2(out2)
#             cost3 = self.classif3(out3)
#
#             cost0 = F.interpolate(cost0, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
#             cost0 = torch.squeeze(cost0, 1)
#             pred0 = F.softmax(cost0, dim=1)
#             pred0 = disparity_regression(pred0, self.maxdisp)
#
#             cost1 = F.interpolate(cost1, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
#             cost1 = torch.squeeze(cost1, 1)
#             pred1 = F.softmax(cost1, dim=1)
#             pred1 = disparity_regression(pred1, self.maxdisp)
#
#             cost2 = F.interpolate(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
#             cost2 = torch.squeeze(cost2, 1)
#             pred2 = F.softmax(cost2, dim=1)
#             pred2 = disparity_regression(pred2, self.maxdisp)
#
#             cost3 = F.interpolate(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
#             cost3 = torch.squeeze(cost3, 1)
#             pred3 = F.softmax(cost3, dim=1)
#             pred3 = disparity_regression(pred3, self.maxdisp)
#             return [pred0, pred1, pred2, pred3]
#
#         else:
#             cost3 = self.classif3(out3)
#             cost3 = F.interpolate(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
#             cost3 = torch.squeeze(cost3, 1)
#             pred3 = F.softmax(cost3, dim=1)
#             pred3 = disparity_regression(pred3, self.maxdisp)
#             return [pred3]
#
#
# def GwcNet_G(d):
#     return GwcNet(d, use_concat_volume=False)
#
#
# def GwcNet_GC(d):
#     return GwcNet(d, use_concat_volume=True)


class GwcNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_network(self, model_cfg):
        print("Building GwcNet.")
        self.maxdisp = model_cfg['max_disp']
        self.use_concat_volume = model_cfg['concat_volume']

        self.num_groups = 40

        if self.use_concat_volume:
            self.concat_channels = 12
            self.feature_extraction = feature_extraction(concat_feature=True,
                                                         concat_feature_channel=self.concat_channels)
        else:
            self.concat_channels = 0
            self.feature_extraction = feature_extraction(concat_feature=False)

        self.dres0 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        print("GwcNet built.")

    def forward(self, data):
        left, right, mask = data['left'], data['right'], data['mask']
        features_left = self.feature_extraction(left)
        features_right = self.feature_extraction(right)

        gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4,
                                      self.num_groups)
        if self.use_concat_volume:
            concat_volume = build_concat_volume(features_left["concat_feature"], features_right["concat_feature"],
                                                self.maxdisp // 4)
            volume = torch.cat((gwc_volume, concat_volume), 1)
        else:
            volume = gwc_volume

        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0

        out1 = self.dres2(cost0)
        out2 = self.dres3(out1)
        out3 = self.dres4(out2)

        if self.training:
            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)
            cost3 = self.classif3(out3)

            cost0 = F.interpolate(cost0, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0, self.maxdisp)

            cost1 = F.interpolate(cost1, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, self.maxdisp)

            cost2 = F.interpolate(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)

            cost3 = F.interpolate(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)
            output = {
                "training_disp": {
                    "disp": pred3,
                    "middle_disp": [pred0, pred1, pred2],
                    "mask": mask,
                    "gt_disp": data['disp']
                },
                "inference_disp": {
                    "disp": pred3,
                },
                "visual_summary": {

                }
            }
            return output

        else:
            cost3 = self.classif3(out3)
            cost3 = F.interpolate(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)
            output = {
                "training_disp": {
                    None
                },
                "inference_disp": {
                    "disp": pred3,
                },
                "visual_summary": {

                }
            }
            return output
