import torch
import torch.nn as nn
import torch.nn.functional as F

from .deform import SimpleBottleneck, DeformSimpleBottleneck


def conv3d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=dilation, dilation=dilation,
                                   bias=False, groups=groups),
                         nn.BatchNorm3d(out_channels),
                         nn.LeakyReLU(0.2, inplace=True))


# Used in PSMNet
def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
        nn.BatchNorm3d(out_planes))


def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=dilation, dilation=dilation,
                                   bias=False, groups=groups),
                         nn.BatchNorm2d(out_channels),
                         nn.LeakyReLU(0.2, inplace=True))


def conv1x1(in_planes, out_planes):
    """1x1 convolution, used for pointwise conv"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                                   padding=0, bias=False),
                         nn.BatchNorm2d(out_planes),
                         nn.LeakyReLU(0.2, inplace=True))


# Used for StereoNet feature extractor
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, with_bn_relu=False):
    """3x3 convolution with padding"""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
    if with_bn_relu:
        conv = nn.Sequential(conv,
                             nn.BatchNorm2d(out_planes),
                             nn.ReLU(inplace=True))
    return conv


# Used for GCNet for aggregation
def conv3x3_3d(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=3,
                                   stride=stride, padding=dilation, dilation=dilation,
                                   groups=groups, bias=False),
                         nn.BatchNorm3d(out_planes),
                         nn.ReLU(inplace=True))


def trans_conv3x3_3d(in_channels, out_channels, stride=1, groups=1, dilation=1):
    return nn.Sequential(nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3,
                                            stride=stride, padding=dilation,
                                            output_padding=dilation,
                                            groups=groups, dilation=dilation,
                                            bias=False),
                         nn.BatchNorm3d(out_channels),
                         nn.ReLU(inplace=True))


class StereoNetAggregation(nn.Module):
    def __init__(self, in_channels=32):
        super(StereoNetAggregation, self).__init__()

        aggregation_modules = nn.ModuleList()

        # StereoNet uses four 3d conv
        for _ in range(4):
            aggregation_modules.append(conv3d(in_channels, in_channels))
        self.aggregation_layer = nn.Sequential(*aggregation_modules)

        self.final_conv = nn.Conv3d(in_channels, 1, kernel_size=3, stride=1,
                                    padding=1, bias=True)

    def forward(self, cost_volume):
        assert cost_volume.dim() == 5  # [B, C, D, H, W]

        out = self.aggregation_layer(cost_volume)
        out = self.final_conv(out)  # [B, 1, D, H, W]
        out = out.squeeze(1)  # [B, D, H, W]

        return out


class PSMNetBasicAggregation(nn.Module):
    """12 3D conv"""

    def __init__(self, max_disp):
        super(PSMNetBasicAggregation, self).__init__()
        self.max_disp = max_disp

        conv0 = convbn_3d(64, 32, 3, 1, 1)
        conv1 = convbn_3d(32, 32, 3, 1, 1)

        final_conv = nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)

        self.dres0 = nn.Sequential(conv0,
                                   nn.ReLU(inplace=True),
                                   conv1,
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(conv1,
                                   nn.ReLU(inplace=True),
                                   conv1)

        self.dres2 = nn.Sequential(conv1,
                                   nn.ReLU(inplace=True),
                                   conv1)

        self.dres3 = nn.Sequential(conv1,
                                   nn.ReLU(inplace=True),
                                   conv1)

        self.dres4 = nn.Sequential(conv1,
                                   nn.ReLU(inplace=True),
                                   conv1)

        self.classify = nn.Sequential(conv1,
                                      nn.ReLU(inplace=True),
                                      final_conv)

    def forward(self, cost):
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0
        cost0 = self.dres2(cost0) + cost0
        cost0 = self.dres3(cost0) + cost0
        cost0 = self.dres4(cost0) + cost0

        cost = self.classify(cost0)  # [B, 1, 48, H/4, W/4]
        cost = F.interpolate(cost, scale_factor=4, mode='trilinear')

        cost = torch.squeeze(cost, 1)  # [B, 192, H, W]

        return [cost]


# PSMNet Hourglass network
class PSMNetHourglass(nn.Module):
    def __init__(self, inplanes):
        super(PSMNetHourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes * 2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes))  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


class PSMNetHGAggregation(nn.Module):
    """22 3D conv"""

    def __init__(self, max_disp):
        super(PSMNetHGAggregation, self).__init__()
        self.max_disp = max_disp

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))  # [B, 32, D/4, H/4, W/4]

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))  # [B, 32, D/4, H/4, W/4]

        self.dres2 = PSMNetHourglass(32)

        self.dres3 = PSMNetHourglass(32)

        self.dres4 = PSMNetHourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

    def forward(self, cost):
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        cost3 = F.interpolate(cost3, scale_factor=4, mode='trilinear')
        cost3 = torch.squeeze(cost3, 1)

        if self.training:
            cost1 = F.interpolate(cost1, scale_factor=4, mode='trilinear')
            cost2 = F.interpolate(cost2, scale_factor=4, mode='trilinear')

            cost1 = torch.squeeze(cost1, 1)
            cost2 = torch.squeeze(cost2, 1)

            return [cost1, cost2, cost3]

        return [cost3]


class GCNetAggregation(nn.Module):
    def __init__(self):
        super(GCNetAggregation, self).__init__()
        self.conv1 = nn.Sequential(conv3x3_3d(64, 32),
                                   conv3x3_3d(32, 32))  # H/2

        self.conv2a = conv3x3_3d(64, 64, stride=2)  # H/4
        self.conv2b = nn.Sequential(conv3x3_3d(64, 64),
                                    conv3x3_3d(64, 64))  # H/4

        self.conv3a = conv3x3_3d(64, 64, stride=2)  # H/8
        self.conv3b = nn.Sequential(conv3x3_3d(64, 64),
                                    conv3x3_3d(64, 64))  # H/8

        self.conv4a = conv3x3_3d(64, 64, stride=2)  # H/16
        self.conv4b = nn.Sequential(conv3x3_3d(64, 64),
                                    conv3x3_3d(64, 64))  # H/16

        self.conv5a = conv3x3_3d(64, 128, stride=2)  # H/32
        self.conv5b = nn.Sequential(conv3x3_3d(128, 128),
                                    conv3x3_3d(128, 128))  # H/32

        self.trans_conv1 = trans_conv3x3_3d(128, 64, stride=2)  # H/16
        self.trans_conv2 = trans_conv3x3_3d(64, 64, stride=2)  # H/8
        self.trans_conv3 = trans_conv3x3_3d(64, 64, stride=2)  # H/4
        self.trans_conv4 = trans_conv3x3_3d(64, 32, stride=2)  # H/2
        self.trans_conv5 = nn.ConvTranspose3d(32, 1, kernel_size=3,
                                              stride=2, padding=1,
                                              groups=1, dilation=1,
                                              bias=False)  # H

    def forward(self, cost_volume):
        conv1 = self.conv1(cost_volume)  # H/2
        conv2a = self.conv2a(cost_volume)  # H/4
        conv2b = self.conv2b(conv2a)  # H/4
        conv3a = self.conv3a(conv2a)  # H/8
        conv3b = self.conv3b(conv3a)  # H/8
        conv4a = self.conv4a(conv3a)  # H/16
        conv4b = self.conv4b(conv4a)  # H/16
        conv5a = self.conv5a(conv4a)  # H/32
        conv5b = self.conv5b(conv5a)  # H/32
        trans_conv1 = self.trans_conv1(conv5b)  # H/16
        trans_conv2 = self.trans_conv2(trans_conv1 + conv4b)  # H/8
        trans_conv3 = self.trans_conv3(trans_conv2 + conv3b)  # H/4
        trans_conv4 = self.trans_conv4(trans_conv3 + conv2b)  # H/2
        trans_conv5 = self.trans_conv5(trans_conv4 + conv1)  # H

        out = torch.squeeze(trans_conv5, 1)  # [B, D, H, W]

        return out


# Adaptive intra-scale aggregation & adaptive cross-scale aggregation
class AdaptiveAggregationModule(nn.Module):
    def __init__(self, num_scales, num_output_branches, max_disp,
                 num_blocks=1,
                 simple_bottleneck=False,
                 deformable_groups=2,
                 mdconv_dilation=2):
        super(AdaptiveAggregationModule, self).__init__()

        self.num_scales = num_scales
        self.num_output_branches = num_output_branches
        self.max_disp = max_disp
        self.num_blocks = num_blocks

        self.branches = nn.ModuleList()

        # Adaptive intra-scale aggregation
        for i in range(self.num_scales):
            num_candidates = max_disp // (2 ** i)
            branch = nn.ModuleList()
            for j in range(num_blocks):
                if simple_bottleneck:
                    branch.append(SimpleBottleneck(num_candidates, num_candidates))
                else:
                    branch.append(DeformSimpleBottleneck(num_candidates, num_candidates, modulation=True,
                                                         mdconv_dilation=mdconv_dilation,
                                                         deformable_groups=deformable_groups))

            self.branches.append(nn.Sequential(*branch))

        self.fuse_layers = nn.ModuleList()

        # Adaptive cross-scale aggregation
        # For each output branch
        for i in range(self.num_output_branches):
            self.fuse_layers.append(nn.ModuleList())
            # For each branch (different scale)
            for j in range(self.num_scales):
                if i == j:
                    # Identity
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    self.fuse_layers[-1].append(
                        nn.Sequential(nn.Conv2d(max_disp // (2 ** j), max_disp // (2 ** i),
                                                kernel_size=1, bias=False),
                                      nn.BatchNorm2d(max_disp // (2 ** i)),
                                      ))
                elif i > j:
                    layers = nn.ModuleList()
                    for k in range(i - j - 1):
                        layers.append(nn.Sequential(nn.Conv2d(max_disp // (2 ** j), max_disp // (2 ** j),
                                                              kernel_size=3, stride=2, padding=1, bias=False),
                                                    nn.BatchNorm2d(max_disp // (2 ** j)),
                                                    nn.LeakyReLU(0.2, inplace=True),
                                                    ))

                    layers.append(nn.Sequential(nn.Conv2d(max_disp // (2 ** j), max_disp // (2 ** i),
                                                          kernel_size=3, stride=2, padding=1, bias=False),
                                                nn.BatchNorm2d(max_disp // (2 ** i))))
                    self.fuse_layers[-1].append(nn.Sequential(*layers))

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        assert len(self.branches) == len(x)

        for i in range(len(self.branches)):
            branch = self.branches[i]
            for j in range(self.num_blocks):
                dconv = branch[j]
                x[i] = dconv(x[i])

        if self.num_scales == 1:  # without fusions
            return x

        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    exchange = self.fuse_layers[i][j](x[j])
                    if exchange.size()[2:] != x_fused[i].size()[2:]:
                        exchange = F.interpolate(exchange, size=x_fused[i].size()[2:],
                                                 mode='bilinear', align_corners=False)
                    x_fused[i] = x_fused[i] + exchange

        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])

        return x_fused


# Stacked AAModules
class AdaptiveAggregation(nn.Module):
    def __init__(self, max_disp, num_scales=3, num_fusions=6,
                 num_stage_blocks=1,
                 num_deform_blocks=2,
                 intermediate_supervision=True,
                 deformable_groups=2,
                 mdconv_dilation=2):
        super(AdaptiveAggregation, self).__init__()

        self.max_disp = max_disp
        self.num_scales = num_scales
        self.num_fusions = num_fusions
        self.intermediate_supervision = intermediate_supervision

        fusions = nn.ModuleList()
        for i in range(num_fusions):
            if self.intermediate_supervision:
                num_out_branches = self.num_scales
            else:
                num_out_branches = 1 if i == num_fusions - 1 else self.num_scales

            if i >= num_fusions - num_deform_blocks:
                simple_bottleneck_module = False
            else:
                simple_bottleneck_module = True

            fusions.append(AdaptiveAggregationModule(num_scales=self.num_scales,
                                                     num_output_branches=num_out_branches,
                                                     max_disp=max_disp,
                                                     num_blocks=num_stage_blocks,
                                                     mdconv_dilation=mdconv_dilation,
                                                     deformable_groups=deformable_groups,
                                                     simple_bottleneck=simple_bottleneck_module))

        self.fusions = nn.Sequential(*fusions)

        self.final_conv = nn.ModuleList()
        for i in range(self.num_scales):
            in_channels = max_disp // (2 ** i)

            self.final_conv.append(nn.Conv2d(in_channels, max_disp // (2 ** i), kernel_size=1))

            if not self.intermediate_supervision:
                break

    def forward(self, cost_volume):
        assert isinstance(cost_volume, list)

        for i in range(self.num_fusions):
            fusion = self.fusions[i]
            cost_volume = fusion(cost_volume)

        # Make sure the final output is in the first position
        out = []  # 1/3, 1/6, 1/12
        for i in range(len(self.final_conv)):
            out = out + [self.final_conv[i](cost_volume[i])]

        return out
