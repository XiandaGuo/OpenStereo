import torch.nn as nn
from functools import partial
from stereo.modeling.common.basic_block_2d import BasicConv2d
from torch.nn.init import kaiming_normal_


class ResBlock(nn.Module):
    def __init__(self, n_in, n_out, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv2d(n_in, n_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class FadnetBackbone(nn.Module):

    def __init__(self, resblock=True, input_channel=3, encoder_ratio=16, decoder_ratio=16):
        super(FadnetBackbone, self).__init__()

        self.input_channel = input_channel
        self.basicC = 2
        self.eratio = encoder_ratio
        self.dratio = decoder_ratio
        self.basicE = self.basicC * self.eratio
        self.basicD = self.basicC * self.dratio

        # shrink and extract features
        self.conv1 = BasicConv2d(in_channels=self.input_channel, out_channels=self.basicE,
                                 norm_layer=None, act_layer=partial(nn.LeakyReLU, negative_slope=0.1, inplace=True),
                                 kernel_size=7, stride=2, padding=3)
        if resblock:
            self.conv2 = ResBlock(self.basicE, 48, stride=2)
            self.conv3 = ResBlock(48, 64, stride=2)
            self.conv4 = ResBlock(64, 192, stride=2)
            self.conv5 = ResBlock(192, 160, stride=2)
        else:
            self.conv2 = BasicConv2d(in_channels=self.basicE, out_channels=self.basicE * 2,
                                     norm_layer=None, act_layer=partial(nn.LeakyReLU, negative_slope=0.1, inplace=True),
                                     kernel_size=3, stride=2, padding=1)
            self.conv3 = BasicConv2d(in_channels=self.basicE * 2, out_channels=self.basicE * 4,
                                     norm_layer=None, act_layer=partial(nn.LeakyReLU, negative_slope=0.1, inplace=True),
                                     kernel_size=3, stride=2, padding=1)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, img):
        conv1_l = self.conv1(img)  # /2
        conv2_l = self.conv2(conv1_l) # /4
        conv3_l = self.conv3(conv2_l) # /8
        conv4_l = self.conv4(conv3_l)  # /16
        conv5_l = self.conv5(conv4_l)  # /32

        return [conv2_l, conv3_l, conv4_l, conv5_l]
