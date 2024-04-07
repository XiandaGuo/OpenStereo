# @Time    : 2023/8/28 00:06
# @Author  : zhangchenming
import torch.nn as nn

class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False,
                 norm_layer=None, act_layer=None, **kwargs):
        super(BasicConv3d, self).__init__()
        layers = [nn.Conv3d(in_channels, out_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, **kwargs)]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if act_layer is not None:
            layers.append(act_layer())

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x


class BasicDeconv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False,
                 norm_layer=None, act_layer=None, **kwargs):
        super(BasicDeconv3d, self).__init__()
        layers = [nn.ConvTranspose3d(in_channels, out_channels,
                                     kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, **kwargs)]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if act_layer is not None:
            layers.append(act_layer())

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x