import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.AANet.deform_conv import DeformConv, ModulatedDeformConv


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


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=dilation, dilation=dilation,
                                   bias=False, groups=groups),
                         nn.BatchNorm2d(out_channels),
                         nn.LeakyReLU(0.2, inplace=True))


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


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AANetFeature(nn.Module):
    def __init__(self, in_channels=32,
                 zero_init_residual=True,
                 groups=1,
                 width_per_group=64,
                 feature_mdconv=True,
                 norm_layer=None):
        super(AANetFeature, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        layers = [3, 4, 6]  # ResNet-40

        # self.inplanes = 64
        self.inplanes = in_channels
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group

        stride = 3
        self.conv1 = nn.Sequential(nn.Conv2d(3, self.inplanes, kernel_size=7, stride=stride,
                                             padding=3, bias=False),
                                   nn.BatchNorm2d(self.inplanes),
                                   nn.ReLU(inplace=True))  # H/3

        self.layer1 = self._make_layer(Bottleneck, in_channels, layers[0])  # H/3
        self.layer2 = self._make_layer(Bottleneck, in_channels * 2, layers[1], stride=2)  # H/6

        block = DeformBottleneck if feature_mdconv else Bottleneck
        self.layer3 = self._make_layer(block, in_channels * 4, layers[2], stride=2)  # H/12

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)

        return [layer1, layer2, layer3]

class DeformConv2d(nn.Module):
    """A single (modulated) deformable conv layer"""

    def __init__(self, in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=2,
                 groups=1,
                 deformable_groups=2,
                 modulation=True,
                 double_mask=True,
                 bias=False):
        super(DeformConv2d, self).__init__()

        self.modulation = modulation
        self.deformable_groups = deformable_groups
        self.kernel_size = kernel_size
        self.double_mask = double_mask

        if self.modulation:
            self.deform_conv = ModulatedDeformConv(in_channels,
                                                   out_channels,
                                                   kernel_size=kernel_size,
                                                   stride=stride,
                                                   padding=dilation,
                                                   dilation=dilation,
                                                   groups=groups,
                                                   deformable_groups=deformable_groups,
                                                   bias=bias)
        else:
            self.deform_conv = DeformConv(in_channels,
                                          out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=dilation,
                                          dilation=dilation,
                                          groups=groups,
                                          deformable_groups=deformable_groups,
                                          bias=bias)

        k = 3 if self.modulation else 2

        offset_out_channels = deformable_groups * k * kernel_size * kernel_size

        # Group-wise offset leraning when deformable_groups > 1
        self.offset_conv = nn.Conv2d(in_channels, offset_out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=dilation, dilation=dilation,
                                     groups=deformable_groups, bias=True)

        # Initialize the weight for offset_conv as 0 to act like regular conv
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

    def forward(self, x):
        if self.modulation:
            offset_mask = self.offset_conv(x)

            offset_channel = self.deformable_groups * 2 * self.kernel_size * self.kernel_size
            offset = offset_mask[:, :offset_channel, :, :]

            mask = offset_mask[:, offset_channel:, :, :]
            mask = mask.sigmoid()  # [0, 1]

            if self.double_mask:
                mask = mask * 2  # initialize as 1 to work as regular conv

            out = self.deform_conv(x, offset, mask)

        else:
            offset = self.offset_conv(x)
            out = self.deform_conv(x, offset)

        return out


class DeformBottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(DeformBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = DeformConv2d(width, width, stride=stride)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SimpleBottleneck(nn.Module):
    """Simple bottleneck block without channel expansion"""

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(SimpleBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DeformSimpleBottleneck(nn.Module):

    """Used for cost aggregation"""

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None,
                 mdconv_dilation=2,
                 deformable_groups=2,
                 modulation=True,
                 double_mask=True,
                 ):
        super(DeformSimpleBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = DeformConv2d(width, width, stride=stride,
                                  dilation=mdconv_dilation,
                                  deformable_groups=deformable_groups,
                                  modulation=modulation,
                                  double_mask=double_mask)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

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


class CostVolume(nn.Module):
    def __init__(self, max_disp, feature_similarity='correlation'):
        """Construct cost volume based on different
        similarity measures

        Args:
            max_disp: max disparity candidate
            feature_similarity: type of similarity measure
        """
        super(CostVolume, self).__init__()

        self.max_disp = max_disp
        self.feature_similarity = feature_similarity

    def forward(self, left_feature, right_feature):
        b, c, h, w = left_feature.size()

        if self.feature_similarity == 'difference':
            cost_volume = left_feature.new_zeros(b, c, self.max_disp, h, w)

            for i in range(self.max_disp):
                if i > 0:
                    cost_volume[:, :, i, :, i:] = left_feature[:, :, :, i:] - right_feature[:, :, :, :-i]
                else:
                    cost_volume[:, :, i, :, :] = left_feature - right_feature

        elif self.feature_similarity == 'concat':
            cost_volume = left_feature.new_zeros(b, 2 * c, self.max_disp, h, w)
            for i in range(self.max_disp):
                if i > 0:
                    cost_volume[:, :, i, :, i:] = torch.cat((left_feature[:, :, :, i:], right_feature[:, :, :, :-i]),
                                                            dim=1)
                else:
                    cost_volume[:, :, i, :, :] = torch.cat((left_feature, right_feature), dim=1)

        elif self.feature_similarity == 'correlation':
            cost_volume = left_feature.new_zeros(b, self.max_disp, h, w)

            for i in range(self.max_disp):
                if i > 0:
                    cost_volume[:, i, :, i:] = (left_feature[:, :, :, i:] *
                                                right_feature[:, :, :, :-i]).mean(dim=1)
                else:
                    cost_volume[:, i, :, :] = (left_feature * right_feature).mean(dim=1)

        else:
            raise NotImplementedError

        cost_volume = cost_volume.contiguous()  # [B, C, D, H, W] or [B, D, H, W]

        return cost_volume


class CostVolumePyramid(nn.Module):
    def __init__(self, max_disp, feature_similarity='correlation'):
        super(CostVolumePyramid, self).__init__()
        self.max_disp = max_disp
        self.feature_similarity = feature_similarity

    def forward(self, left_feature_pyramid, right_feature_pyramid):
        num_scales = len(left_feature_pyramid)

        cost_volume_pyramid = []
        for s in range(num_scales):
            max_disp = self.max_disp // (2 ** s)
            cost_volume_module = CostVolume(max_disp, self.feature_similarity)
            cost_volume = cost_volume_module(left_feature_pyramid[s],
                                             right_feature_pyramid[s])
            cost_volume_pyramid.append(cost_volume)

        return cost_volume_pyramid  # H/3, H/6, H/12


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


class DisparityEstimation(nn.Module):
    def __init__(self, max_disp, match_similarity=True):
        super(DisparityEstimation, self).__init__()

        self.max_disp = max_disp
        self.match_similarity = match_similarity

    def forward(self, cost_volume):
        assert cost_volume.dim() == 4

        # Matching similarity or matching cost
        cost_volume = cost_volume if self.match_similarity else -cost_volume

        prob_volume = F.softmax(cost_volume, dim=1)  # [B, D, H, W]

        if cost_volume.size(1) == self.max_disp:
            disp_candidates = torch.arange(0, self.max_disp).type_as(prob_volume)
        else:
            max_disp = prob_volume.size(1)  # current max disparity
            disp_candidates = torch.arange(0, max_disp).type_as(prob_volume)

        disp_candidates = disp_candidates.view(1, cost_volume.size(1), 1, 1)
        disp = torch.sum(prob_volume * disp_candidates, 1, keepdim=False)  # [B, H, W]

        return disp

class StereoDRNetRefinement(nn.Module):
    def __init__(self):
        super(StereoDRNetRefinement, self).__init__()

        # Left and warped error
        in_channels = 6

        self.conv1 = conv2d(in_channels, 16)
        self.conv2 = conv2d(1, 16)  # on low disparity

        self.dilation_list = [1, 2, 4, 8, 1, 1]
        self.dilated_blocks = nn.ModuleList()

        for dilation in self.dilation_list:
            self.dilated_blocks.append(BasicBlock(32, 32, stride=1, dilation=dilation))

        self.dilated_blocks = nn.Sequential(*self.dilated_blocks)

        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, low_disp, left_img, right_img):
        assert low_disp.dim() == 3
        low_disp = low_disp.unsqueeze(1)  # [B, 1, H, W]
        scale_factor = left_img.size(-1) / low_disp.size(-1)
        if scale_factor == 1.0:
            disp = low_disp
        else:
            disp = F.interpolate(low_disp, size=left_img.size()[-2:], mode='bilinear', align_corners=False)
            disp = disp * scale_factor

        # Warp right image to left view with current disparity
        warped_right = disp_warp(right_img, disp)[0]  # [B, C, H, W]
        error = warped_right - left_img  # [B, C, H, W]

        concat1 = torch.cat((error, left_img), dim=1)  # [B, 6, H, W]

        conv1 = self.conv1(concat1)  # [B, 16, H, W]
        conv2 = self.conv2(disp)  # [B, 16, H, W]
        concat2 = torch.cat((conv1, conv2), dim=1)  # [B, 32, H, W]

        out = self.dilated_blocks(concat2)  # [B, 32, H, W]
        residual_disp = self.final_conv(out)  # [B, 1, H, W]

        disp = F.relu(disp + residual_disp, inplace=True)  # [B, 1, H, W]
        disp = disp.squeeze(1)  # [B, H, W]

        return disp

def normalize_coords(grid):
    """Normalize coordinates of image scale to [-1, 1]
    Args:
        grid: [B, 2, H, W]
    """
    assert grid.size(1) == 2
    h, w = grid.size()[2:]
    grid[:, 0, :, :] = 2 * (grid[:, 0, :, :].clone() / (w - 1)) - 1  # x: [-1, 1]
    grid[:, 1, :, :] = 2 * (grid[:, 1, :, :].clone() / (h - 1)) - 1  # y: [-1, 1]
    grid = grid.permute((0, 2, 3, 1))  # [B, H, W, 2]
    return grid


def meshgrid(img, homogeneous=False):
    """Generate meshgrid in image scale
    Args:
        img: [B, _, H, W]
        homogeneous: whether to return homogeneous coordinates
    Return:
        grid: [B, 2, H, W]
    """
    b, _, h, w = img.size()

    x_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(img)  # [1, H, W]
    y_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(img)

    grid = torch.cat((x_range, y_range), dim=0)  # [2, H, W], grid[:, i, j] = [j, i]
    grid = grid.unsqueeze(0).expand(b, 2, h, w)  # [B, 2, H, W]

    if homogeneous:
        ones = torch.ones_like(x_range).unsqueeze(0).expand(b, 1, h, w)  # [B, 1, H, W]
        grid = torch.cat((grid, ones), dim=1)  # [B, 3, H, W]
        assert grid.size(1) == 3
    return grid


def disp_warp(img, disp, padding_mode='border'):
    """Warping by disparity
    Args:
        img: [B, 3, H, W]
        disp: [B, 1, H, W], positive
        padding_mode: 'zeros' or 'border'
    Returns:
        warped_img: [B, 3, H, W]
        valid_mask: [B, 3, H, W]
    """
    assert disp.min() >= 0

    grid = meshgrid(img)  # [B, 2, H, W] in image scale
    # Note that -disp here
    offset = torch.cat((-disp, torch.zeros_like(disp)), dim=1)  # [B, 2, H, W]
    sample_grid = grid + offset
    sample_grid = normalize_coords(sample_grid)  # [B, H, W, 2] in [-1, 1]
    warped_img = F.grid_sample(img, sample_grid, mode='bilinear', padding_mode=padding_mode)

    mask = torch.ones_like(img)
    valid_mask = F.grid_sample(mask, sample_grid, mode='bilinear', padding_mode='zeros')
    valid_mask[valid_mask < 0.9999] = 0
    valid_mask[valid_mask > 0] = 1
    return warped_img, valid_mask


