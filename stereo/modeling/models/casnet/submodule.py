import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

Align_Corners = False
Align_Corners_Range = False


def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.BatchNorm3d(out_channels))


def disparity_regression(x, disp_values):
    assert len(x.shape) == 4
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


def get_cur_disp_range_samples(cur_disp, ndisp, disp_inteval_pixel, shape, ns_size, using_ns=False, max_disp=192.0):
    # shape, (B, H, W)
    # cur_disp: (B, H, W)
    # return disp_range_samples: (B, D, H, W)
    if not using_ns:
        cur_disp_min = (cur_disp - ndisp / 2 * disp_inteval_pixel)  # (B, H, W)
        cur_disp_max = (cur_disp + ndisp / 2 * disp_inteval_pixel)
        # cur_disp_min = (cur_disp - ndisp / 2 * disp_inteval_pixel).clamp(min=0.0)   #(B, H, W)
        # cur_disp_max = (cur_disp_min + (ndisp - 1) * disp_inteval_pixel).clamp(max=max_disp)

        assert cur_disp.shape == torch.Size(shape), "cur_disp:{}, input shape:{}".format(cur_disp.shape, shape)
        new_interval = (cur_disp_max - cur_disp_min) / (ndisp - 1)  # (B, H, W)

        disp_range_samples = cur_disp_min.unsqueeze(1) + (torch.arange(0, ndisp, device=cur_disp.device,
                                                                       dtype=cur_disp.dtype,
                                                                       requires_grad=False).reshape(1, -1, 1,
                                                                                                    1) * new_interval.unsqueeze(
            1))
    else:
        # using neighbor region information to help determine the range.
        # consider the maximum and minimum values ​​in the region.
        assert cur_disp.shape == torch.Size(shape), "cur_disp:{}, input shape:{}".format(cur_disp.shape, shape)
        B, H, W = cur_disp.shape
        cur_disp_smooth = F.interpolate((cur_disp / 4.0).unsqueeze(1),
                                        [H // 4, W // 4], mode='bilinear', align_corners=Align_Corners_Range).squeeze(1)
        # get minimum value
        disp_min_ns = torch.abs(
            F.max_pool2d(-cur_disp_smooth, stride=1, kernel_size=ns_size, padding=ns_size // 2))  # (B, 1/4H, 1/4W)
        # get maximum value
        disp_max_ns = F.max_pool2d(cur_disp_smooth, stride=1, kernel_size=ns_size, padding=ns_size // 2)

        disp_pred_inter = torch.abs(disp_max_ns - disp_min_ns)  # (B, 1/4H, 1/4W)
        disp_range_comp = (ndisp // 4 * disp_inteval_pixel - disp_pred_inter).clamp(min=0) / 2.0  # (B, 1/4H, 1/4W)

        cur_disp_min = (disp_min_ns - disp_range_comp).clamp(min=0, max=max_disp)
        cur_disp_max = (disp_max_ns + disp_range_comp).clamp(min=0, max=max_disp)

        new_interval = (cur_disp_max - cur_disp_min) / (ndisp // 4 - 1)  # (B, 1/4H, 1/4W)

        # (B, 1/4D, 1/4H, 1/4W)
        disp_range_samples = cur_disp_min.unsqueeze(1) + (torch.arange(0, ndisp // 4, device=cur_disp.device,
                                                                       dtype=cur_disp.dtype,
                                                                       requires_grad=False).reshape(1, -1, 1,
                                                                                                    1) * new_interval.unsqueeze(
            1))
        # (B, D, H, W)
        disp_range_samples = F.interpolate((disp_range_samples * 4.0).unsqueeze(1),
                                           [ndisp, H, W], mode='trilinear', align_corners=Align_Corners_Range).squeeze(
            1)
    return disp_range_samples


def get_disp_range_samples(cur_disp, ndisp, disp_inteval_pixel, device, dtype, shape, using_ns, ns_size,
                           max_disp=192.0):
    # shape, (B, H, W)
    # cur_disp: (B, H, W) or float
    # return disp_range_values: (B, D, H, W)
    # with torch.no_grad():
    if cur_disp is None:
        cur_disp = torch.tensor(0, device=device, dtype=dtype, requires_grad=False).reshape(1, 1, 1).repeat(*shape)
        cur_disp_min = (cur_disp - ndisp / 2 * disp_inteval_pixel).clamp(min=0.0)  # (B, H, W)
        cur_disp_max = (cur_disp_min + (ndisp - 1) * disp_inteval_pixel).clamp(max=max_disp)
        new_interval = (cur_disp_max - cur_disp_min) / (ndisp - 1)  # (B, H, W)

        disp_range_volume = cur_disp_min.unsqueeze(1) + (torch.arange(0, ndisp, device=cur_disp.device,
                                                                      dtype=cur_disp.dtype,
                                                                      requires_grad=False).reshape(1, -1, 1,
                                                                                                   1) * new_interval.unsqueeze(
            1))

    else:
        disp_range_volume = get_cur_disp_range_samples(cur_disp, ndisp, disp_inteval_pixel, shape, ns_size, using_ns,
                                                       max_disp)

    return disp_range_volume
