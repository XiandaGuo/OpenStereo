import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class pyramidPooling(nn.Module):

    def __init__(self, in_channels, pool_sizes, model_name='pspnet', fusion_mode='cat', with_bn=True):
        super(pyramidPooling, self).__init__()

        bias = not with_bn

        self.paths = []
        if pool_sizes is None:
            for i in range(4):
                self.paths.append(conv2DBatchNormRelu(in_channels, in_channels, 1, 1, 0, bias=bias, with_bn=with_bn))
        else:
            for i in range(len(pool_sizes)):
                self.paths.append(
                    conv2DBatchNormRelu(in_channels, int(in_channels / len(pool_sizes)), 1, 1, 0, bias=bias,
                                        with_bn=with_bn))

        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes
        self.model_name = model_name
        self.fusion_mode = fusion_mode

    # @profile
    def forward(self, x):
        h, w = x.shape[2:]

        k_sizes = []
        strides = []
        if self.pool_sizes is None:
            for pool_size in np.linspace(2, min(h, w), 4, dtype=int):
                k_sizes.append((int(h / pool_size), int(w / pool_size)))
                strides.append((int(h / pool_size), int(w / pool_size)))
            k_sizes = k_sizes[::-1]
            strides = strides[::-1]
        else:
            k_sizes = [(self.pool_sizes[0], self.pool_sizes[0]), (self.pool_sizes[1], self.pool_sizes[1]),
                       (self.pool_sizes[2], self.pool_sizes[2]), (self.pool_sizes[3], self.pool_sizes[3])]
            strides = k_sizes

        if self.fusion_mode == 'cat':  # pspnet: concat (including x)
            output_slices = [x]

            for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                # out = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))
                if self.model_name != 'icnet':
                    out = module(out)
                out = F.upsample(out, size=(h, w), mode='bilinear')
                output_slices.append(out)

            return torch.cat(output_slices, dim=1)
        else:  # icnet: element-wise sum (including x)
            pp_sum = x

            for i, module in enumerate(self.path_module_list):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                out = module(out)
                out = F.upsample(out, size=(h, w), mode='bilinear')
                pp_sum = pp_sum + 0.25 * out
            # pp_sum = F.relu(pp_sum / 2., inplace=True)
            pp_sum = FMish(pp_sum / 2.)

            return pp_sum


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, with_bn=True):
        super(conv2DBatchNormRelu, self).__init__()

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=1)

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          Mish())
        else:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          Mish())

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        # print("Mish activation loaded...")

    def forward(self, x):
        # save 1 second per epoch with no x= x*() and then return x...just inline it.
        return x * (torch.tanh(F.softplus(x)))


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


def disparity_variance(x, maxdisp, disparity):
    # the shape of disparity should be B,1,H,W, return is the variance of the cost volume [B,1,H,W]
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    disp_values = (disp_values - disparity) ** 2
    return torch.sum(x * disp_values, 1, keepdim=True)


def disparity_variance_confidence(x, disparity_samples, disparity):
    # the shape of disparity should be B,1,H,W, return is the uncertainty estimation
    assert len(x.shape) == 4
    disp_values = (disparity - disparity_samples) ** 2
    return torch.sum(x * disp_values, 1, keepdim=True)


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


def groupwise_correlation_4D(fea1, fea2, num_groups):
    B, C, D, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, D, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, D, H, W)
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


def build_corrleation_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, 2 * maxdisp + 1, H, W])
    for i in range(-maxdisp, maxdisp + 1):
        if i > 0:
            volume[:, :, i + maxdisp, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:],
                                                                     targetimg_fea[:, :, :, :-i],
                                                                     num_groups)
        elif i < 0:
            volume[:, :, i + maxdisp, :, :-i] = groupwise_correlation(refimg_fea[:, :, :, :-i],
                                                                      targetimg_fea[:, :, :, i:],
                                                                      num_groups)
        else:
            volume[:, :, i + maxdisp, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


def warp(x, disp):
    """
    warp an image/tensor (imright) back to imleft, according to the disp

    x: [B, C, H, W] (imright)
    disp: [B, 1, H, W] disp

    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    # xx = xx.float()
    # yy = yy.float()
    # grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        xx = xx.float().cuda()
        yy = yy.float().cuda()
    xx_warp = Variable(xx) - disp
    yy = Variable(yy)
    vgrid = torch.cat((xx_warp, yy), 1)
    # vgrid = Variable(grid) + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    # grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0.
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    # grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0.
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output * mask


def FMish(x):
    '''

    Applies the mish function element-wise:

    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

    See additional documentation for mish class.

    '''

    return x * torch.tanh(F.softplus(x))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   Mish())

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride
        # self.gc = ContextBlock2d(planes, planes // 8, 'att', ['channel_add'])

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        # out = self.gc(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class UniformSampler(nn.Module):
    def __init__(self):
        super(UniformSampler, self).__init__()

    def forward(self, min_disparity, max_disparity, number_of_samples=10):
        """
        Args:
            :min_disparity: lower bound of disparity search range
            :max_disparity: upper bound of disparity range predictor
            :number_of_samples (default:10): number of samples to be genearted.
        Returns:
            :sampled_disparities: Uniformly generated disparity samples from the input search range.
        """

        device = min_disparity.get_device()

        multiplier = (max_disparity - min_disparity) / (number_of_samples + 1)  # B,1,H,W
        range_multiplier = torch.arange(1.0, number_of_samples + 1, 1, device=device).view(number_of_samples, 1,
                                                                                           1)  # (number_of_samples, 1, 1)
        sampled_disparities = min_disparity + multiplier * range_multiplier

        return sampled_disparities


class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()

    def forward(self, left_input, right_input, disparity_samples):
        """
        Disparity Sample Cost Evaluator
        Description:
                Given the left image features, right iamge features and the disparity samples, generates:
                    - Warped right image features

        Args:
            :left_input: Left Image Features
            :right_input: Right Image Features
            :disparity_samples:  Disparity Samples

        Returns:
            :warped_right_feature_map: right iamge features warped according to input disparity.
            :left_feature_map: expanded left image features.
        """

        device = left_input.get_device()
        left_y_coordinate = torch.arange(0.0, left_input.size()[3], device=device).repeat(left_input.size()[2])
        left_y_coordinate = left_y_coordinate.view(left_input.size()[2], left_input.size()[3])
        left_y_coordinate = torch.clamp(left_y_coordinate, min=0, max=left_input.size()[3] - 1)
        left_y_coordinate = left_y_coordinate.expand(left_input.size()[0], -1, -1)

        right_feature_map = right_input.expand(disparity_samples.size()[1], -1, -1, -1, -1).permute([1, 2, 0, 3, 4])
        left_feature_map = left_input.expand(disparity_samples.size()[1], -1, -1, -1, -1).permute([1, 2, 0, 3, 4])

        disparity_samples = disparity_samples.float()

        right_y_coordinate = left_y_coordinate.expand(
            disparity_samples.size()[1], -1, -1, -1).permute([1, 0, 2, 3]) - disparity_samples

        right_y_coordinate_1 = right_y_coordinate
        right_y_coordinate = torch.clamp(right_y_coordinate, min=0, max=right_input.size()[3] - 1)

        warped_right_feature_map = torch.gather(right_feature_map, dim=4, index=right_y_coordinate.expand(
            right_input.size()[1], -1, -1, -1, -1).permute([1, 0, 2, 3, 4]).long())

        right_y_coordinate_1 = right_y_coordinate_1.unsqueeze(1)
        warped_right_feature_map = (1 - ((right_y_coordinate_1 < 0) +
                                         (right_y_coordinate_1 > right_input.size()[3] - 1)).float()) * \
                                   (warped_right_feature_map) + torch.zeros_like(warped_right_feature_map)

        return warped_right_feature_map, left_feature_map
