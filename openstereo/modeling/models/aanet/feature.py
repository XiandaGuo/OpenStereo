import torch
import torch.nn as nn
import torch.nn.functional as F

from .deform import DeformConv2d


def conv1x1(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(inplace=True))


# Used for StereoNet feature extractor
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, with_bn_relu=False, leaky_relu=False):
    """3x3 convolution with padding"""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
    if with_bn_relu:
        relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
        conv = nn.Sequential(conv,
                             nn.BatchNorm2d(out_planes),
                             relu)
    return conv


def conv5x5(in_channels, out_channels, stride=2,
            dilation=1, use_bn=True):
    bias = False if use_bn else True
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride,
                     padding=2, dilation=dilation, bias=bias)
    relu = nn.ReLU(inplace=True)
    if use_bn:
        out = nn.Sequential(conv,
                            nn.BatchNorm2d(out_channels),
                            relu)
    else:
        out = nn.Sequential(conv, relu)
    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, leaky_relu=True):
        """StereoNet uses leaky relu (alpha = 0.2)"""
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class StereoNetFeature(nn.Module):
    def __init__(self, num_downsample=3):
        """Feature extractor of StereoNet
        Args:
            num_downsample: 2, 3 or 4
        """
        super(StereoNetFeature, self).__init__()

        self.num_downsample = num_downsample

        downsample = nn.ModuleList()

        in_channels = 3
        out_channels = 32
        for _ in range(num_downsample):
            downsample.append(conv5x5(in_channels, out_channels))
            in_channels = 32

        self.downsample = nn.Sequential(*downsample)

        residual_blocks = nn.ModuleList()

        for _ in range(6):
            residual_blocks.append(BasicBlock(out_channels, out_channels))
        self.residual_blocks = nn.Sequential(*residual_blocks)

        # StereoNet has no bn and relu for last conv layer,
        self.final_conv = conv3x3(out_channels, out_channels)

    def forward(self, img):
        out = self.downsample(img)  # [B, 32, H/8, W/8]
        out = self.residual_blocks(out)  # [B, 32, H/8, W/8]
        out = self.final_conv(out)  # [B, 32, H/8, W/8]

        return out


# Used for PSMNet feature extractor
def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


class PSMNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(PSMNetBasicBlock, self).__init__()

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


class FeaturePyrmaid(nn.Module):
    def __init__(self, in_channel=32):
        super(FeaturePyrmaid, self).__init__()

        self.out1 = nn.Sequential(nn.Conv2d(in_channel, in_channel * 2, kernel_size=3,
                                            stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(in_channel * 2),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(in_channel * 2, in_channel * 2, kernel_size=1,
                                            stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(in_channel * 2),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  )

        self.out2 = nn.Sequential(nn.Conv2d(in_channel * 2, in_channel * 4, kernel_size=3,
                                            stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(in_channel * 4),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(in_channel * 4, in_channel * 4, kernel_size=1,
                                            stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(in_channel * 4),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  )

    def forward(self, x):
        # x: [B, 32, H, W]
        out1 = self.out1(x)  # [B, 64, H/2, W/2]
        out2 = self.out2(out1)  # [B, 128, H/4, W/4]

        return [x, out1, out2]


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels, out_channels=128,
                 num_levels=3):
        # FPN paper uses 256 out channels by default
        super(FeaturePyramidNetwork, self).__init__()

        assert isinstance(in_channels, list)

        self.in_channels = in_channels

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(num_levels):
            lateral_conv = nn.Conv2d(in_channels[i], out_channels, 1)
            fpn_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))

            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        # Inputs: resolution high -> low
        assert len(self.in_channels) == len(inputs)

        # Build laterals
        laterals = [lateral_conv(inputs[i])
                    for i, lateral_conv in enumerate(self.lateral_convs)]

        # Build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        # Build outputs
        out = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        return out


class PSMNetFeature(nn.Module):
    def __init__(self):
        super(PSMNetFeature, self).__init__()
        self.inplanes = 32

        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))  # H/2

        self.layer1 = self._make_layer(PSMNetBasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(PSMNetBasicBlock, 64, 16, 2, 1, 1)  # H/4
        self.layer3 = self._make_layer(PSMNetBasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(PSMNetBasicBlock, 128, 3, 1, 1, 2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))

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
        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.interpolate(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear', align_corners=False)

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.interpolate(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear', align_corners=False)

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.interpolate(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear', align_corners=False)

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.interpolate(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear', align_corners=False)

        output_feature = torch.cat(
            (output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)  # [32, H/4, W/4]

        return output_feature


# GANet feature
class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, bn=True, relu=True,
                 mdconv=False):
        super(Conv2x, self).__init__()
        self.concat = concat

        if deconv and is_3d:
            kernel = (3, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3
        self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel,
                               stride=2, padding=1)

        if self.concat:
            if mdconv:
                self.conv2 = DeformConv2d(out_channels * 2, out_channels, kernel_size=3, stride=1)
            else:
                self.conv2 = BasicConv(out_channels * 2, out_channels, False, is_3d, bn, relu, kernel_size=3,
                                       stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1,
                                   padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        assert (x.size() == rem.size())
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


class GANetFeature(nn.Module):
    """Height and width need to be divided by 48, downsampled by 1/3"""

    def __init__(self, feature_mdconv=False):
        super(GANetFeature, self).__init__()

        if feature_mdconv:
            self.conv_start = nn.Sequential(
                BasicConv(3, 32, kernel_size=3, padding=1),
                BasicConv(32, 32, kernel_size=5, stride=3, padding=2),
                DeformConv2d(32, 32))
        else:
            self.conv_start = nn.Sequential(
                BasicConv(3, 32, kernel_size=3, padding=1),
                BasicConv(32, 32, kernel_size=5, stride=3, padding=2),
                BasicConv(32, 32, kernel_size=3, padding=1))

        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)

        if feature_mdconv:
            self.conv3a = DeformConv2d(64, 96, kernel_size=3, stride=2)
            self.conv4a = DeformConv2d(96, 128, kernel_size=3, stride=2)
        else:
            self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)
            self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, padding=1)

        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)

        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)

        if feature_mdconv:
            self.conv3b = Conv2x(64, 96, mdconv=True)
            self.conv4b = Conv2x(96, 128, mdconv=True)
        else:
            self.conv3b = Conv2x(64, 96)
            self.conv4b = Conv2x(96, 128)

        self.deconv4b = Conv2x(128, 96, deconv=True)
        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.deconv1b = Conv2x(48, 32, deconv=True)

    def forward(self, x):
        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x
        x = self.deconv4a(x, rem3)
        rem3 = x

        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)

        x = self.deconv4b(x, rem3)
        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)  # [B, 32, H/3, W/3]

        return x


class GCNetFeature(nn.Module):
    def __init__(self):
        super(GCNetFeature, self).__init__()

        self.inplanes = 32
        self.conv1 = conv5x5(3, 32)
        self.conv2 = self._make_layer(PSMNetBasicBlock, 32, 8, 1, 1, 1)
        self.conv3 = conv3x3(32, 32)

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
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # [32, H/2, W/2]

        return x
