import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.GANet.modules.GANet import DisparityRegression, GetCostVolume
from libs.GANet.modules.GANet import SGA
from libs.GANet.modules.GANet import NLFIter
from libs.GANet.modules.GANet import GetWeights, GetFilters


class DomainNorm2(nn.Module):
    def __init__(self, channel, l2=True):
        super().__init__()
        self.normalize = nn.InstanceNorm2d(num_features=channel, affine=False)
        self.l2 = l2
        self.weight = nn.Parameter(torch.ones(1, channel, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.weight.requires_grad = True
        self.bias.requires_grad = True

    def forward(self, x):
        x = self.normalize(x)
        if self.l2:
            x = F.normalize(x, p=2, dim=1)
        return x * self.weight + self.bias


class DomainNorm(nn.Module):
    def __init__(self, channel, l2=True):
        super(DomainNorm, self).__init__()
        self.normalize = nn.InstanceNorm2d(num_features=channel, affine=True)
        self.l2 = l2

    def forward(self, x):
        if self.l2:
            x = F.normalize(x, p=2, dim=1)
        x = self.normalize(x)
        return x


class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, l2=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
        #        print(in_channels, out_channels, deconv, is_3d, bn, relu, kwargs)
        self.relu = relu
        self.use_bn = bn
        self.l2 = l2
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
            self.bn = DomainNorm(channel=out_channels, l2=self.l2)

    #            self.bn = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, bn=True, relu=True):
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
            self.conv2 = BasicConv(out_channels * 2, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1,
                                   padding=1)
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


class Pyramid(nn.Module):

    def __init__(self, in_channel=32, inner_channel=24, out_channel=32):
        super(Pyramid, self).__init__()
        self.conv1 = nn.Sequential(nn.AvgPool2d((2, 2), stride=(2, 2)),
                                   nn.Conv2d(in_channel, inner_channel, (3, 3), (1, 1), (1, 1), bias=False))
        self.conv2 = nn.Sequential(nn.AvgPool2d((4, 4), stride=(4, 4)),
                                   nn.Conv2d(in_channel, inner_channel, (3, 3), (1, 1), (1, 1), bias=False))
        self.conv3 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                   nn.Conv2d(in_channel, inner_channel, (3, 3), (1, 1), (1, 1), bias=False))
        self.conv4 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                   nn.Conv2d(in_channel, inner_channel, (3, 3), (1, 1), (1, 1), bias=False))
        self.conv0 = nn.Conv2d(in_channel, in_channel, (3, 3), (1, 1), (1, 1), bias=False)
        self.bn_relu = nn.Sequential(DomainNorm(inner_channel * 4 + in_channel),
                                     nn.ReLU(inplace=True))
        self.conv_last = BasicConv(inner_channel * 4 + in_channel, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x0):
        x0 = self.conv0(x0)
        x1 = F.interpolate(self.conv1(x0), [x0.size()[2], x0.size()[3]], mode='bilinear', align_corners=False)
        x2 = F.interpolate(self.conv2(x0), [x0.size()[2], x0.size()[3]], mode='bilinear', align_corners=False)
        x3 = F.interpolate(self.conv3(x0), [x0.size()[2], x0.size()[3]], mode='bilinear', align_corners=False)
        x4 = F.interpolate(self.conv4(x0), [x0.size()[2], x0.size()[3]], mode='bilinear', align_corners=False)
        x = torch.cat((x0, x1, x2, x3, x4), 1)
        x = self.conv_last(self.bn_relu(x))
        return x


class NLFilter(nn.Module):

    def __init__(self, in_channel=32):
        super(NLFilter, self).__init__()
        self.inner_channel = in_channel
        self.conv_refine = BasicConv(in_channel + self.inner_channel, in_channel, bn=True, relu=True, kernel_size=3,
                                     stride=1, padding=1)
        self.getweights = nn.Sequential(
            BasicConv(in_channel, self.inner_channel, bn=True, l2=False, relu=False, kernel_size=3, stride=1,
                      padding=1),
            GetWeights())
        #        self.amplify = nn.Conv2d(5, 5, kernel_size=1, stride=1, padding=0, bias=False)
        #        self.softmax = nn.Softmax(dim=1)
        self.nlf = NLFIter()

    def forward(self, x):
        rem = x

        k1, k2, k3, k4 = self.getweights(x)
        #        k1 = F.normalize(k1, p=1, dim=1)
        #        k2 = F.normalize(k2, p=1, dim=1)
        #        k3 = F.normalize(k3, p=1, dim=1)
        #        k4 = F.normalize(k4, p=1, dim=1)

        x = self.nlf(x, k1, k2, k3, k4)

        x = torch.cat((x, rem), 1).contiguous()
        x = self.conv_refine(x)
        return x


class NLFilter3d(nn.Module):
    def __init__(self, in_channel=32):
        super(NLFilter3d, self).__init__()
        self.inner_channel = in_channel
        self.conv = BasicConv(in_channel, self.inner_channel, bn=True, relu=False, kernel_size=3, stride=1, padding=1)
        self.conv_refine = BasicConv(in_channel + self.inner_channel, in_channel, is_3d=True, bn=True, relu=True,
                                     kernel_size=3, stride=1, padding=1)
        self.getweights = GetWeights()
        self.nlf = NLFIter()

    def forward(self, x, g):
        rem = x
        g = self.conv(g)
        g0, g1, g2, g3 = self.getweights(g)
        x = self.nlf(x, g0, g1, g2, g3)

        assert (x.size() == rem.size())
        x = torch.cat((x, rem), 1).contiguous()
        x = self.conv_refine(x)
        return x


class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()

        self.conv_start = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, padding=1),
            BasicConv(32, 32, kernel_size=5, stride=2, padding=1),
            BasicConv(32, 32, kernel_size=3, stride=2, padding=1),
            BasicConv(32, 32, relu=False, kernel_size=3, padding=1))
        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)
        self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, padding=1)

        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)

        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96)
        self.conv4b = Conv2x(96, 128)

        self.deconv4b = Conv2x(128, 96, deconv=True)
        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.deconv1b = Conv2x(48, 32, deconv=True)

        self.nlf1 = NLFilter(in_channel=32)
        #        self.refine1a = BasicConv(32, 32, relu=False, kernel_size=3, stride=1, padding=1)
        self.nlf11 = NLFilter(in_channel=48)
        #        self.refine1b = BasicConv(48, 48, relu=False, kernel_size=3, stride=1, padding=1)
        self.nlf12 = NLFilter(in_channel=48)
        #        self.refine1c = BasicConv(48, 48, relu=False, kernel_size=3, stride=1, padding=1)
        self.nlf2 = NLFilter(in_channel=32)
        #        self.refine2a = BasicConv(32, 32, relu=False, kernel_size=3, stride=1, padding=1)
        self.nlf13 = NLFilter(in_channel=48)
        #        self.refine2b = BasicConv(48, 48, relu=False, kernel_size=3, stride=1, padding=1)
        self.nlf14 = NLFilter(in_channel=48)
        #        self.refine2c = BasicConv(48, 48, relu=False, kernel_size=3, stride=1, padding=1)
        self.nlf3 = NLFilter(in_channel=32)
        #        self.refine = BasicConv(32, 32, relu=False, kernel_size=3, stride=1, padding=1)
        self.pyramid1 = Pyramid()
        self.pyramid2 = Pyramid()

    #        self.getweights = nn.Sequential(GetFilters(radius=1),
    #                                        nn.Conv2d(9, 20, kernel_size=1, stride=1, padding=0, bias=False))
    def forward(self, x):
        x = self.conv_start(x)

        x = self.nlf1(x)
        rem0 = x
        x = self.conv1a(x)
        x = self.nlf11(x)
        #       return x
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

        x = self.nlf12(x)
        rem1 = x
        x = self.deconv1a(x, rem0)

        x = self.nlf2(x)

        x = self.pyramid1(x)
        rem0 = x

        x = self.conv1b(x, rem1)

        x = self.nlf13(x)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)

        x = self.deconv4b(x, rem3)
        #        rem3 = x
        x = self.deconv3b(x, rem2)
        #        rem2 = x
        x = self.deconv2b(x, rem1)

        x = self.nlf14(x)
        #        rem1 = x
        x = self.deconv1b(x, rem0)

        x = self.nlf3(x)
        x = self.pyramid2(x)
        return x


#        return self.attention1d(x)
#        return self.pyramid(x, rem1, rem2, rem3)


class Guidance(nn.Module):
    def __init__(self):
        super(Guidance, self).__init__()

        self.conv0 = BasicConv(64, 32, kernel_size=3, padding=1)
        self.conv1 = nn.Sequential(
            BasicConv(32, 32, kernel_size=5, stride=2, padding=1),
            BasicConv(32, 32, kernel_size=3, stride=2, padding=1),
            BasicConv(32, 32, kernel_size=3, padding=1))

        #        self.conv11 = Conv2x(32, 48)

        #        self.conv1 = nn.Sequential(BasicConv(32, 32, kernel_size=3, padding=1),
        #                                   BasicConv(32, 32, kernel_size=3, padding=1))
        self.conv2 = BasicConv(32, 32, kernel_size=3, padding=1)
        self.conv3 = BasicConv(32, 32, kernel_size=3, padding=1)

        #        self.conv11 = Conv2x(32, 48)
        self.conv11 = nn.Sequential(BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
                                    BasicConv(48, 48, kernel_size=3, padding=1))
        self.conv12 = BasicConv(48, 48, kernel_size=3, padding=1)
        self.conv13 = BasicConv(48, 48, kernel_size=3, padding=1)
        self.conv14 = BasicConv(48, 48, kernel_size=3, padding=1)
        self.weight_sg1 = BasicConv(32, 32, relu=False, kernel_size=3, padding=1)
        self.weight_sg2 = BasicConv(32, 32, relu=False, kernel_size=3, padding=1)
        self.weight_sg3 = BasicConv(32, 32, relu=False, kernel_size=3, padding=1)

        self.weight_sg11 = BasicConv(48, 48, relu=False, kernel_size=3, padding=1)
        self.weight_sg12 = BasicConv(48, 48, relu=False, kernel_size=3, padding=1)
        self.weight_sg13 = BasicConv(48, 48, relu=False, kernel_size=3, padding=1)
        self.weight_sg14 = BasicConv(48, 48, relu=False, kernel_size=3, padding=1)

        self.getweights = nn.Sequential(GetFilters(radius=1),
                                        nn.Conv2d(9, 20, kernel_size=1, stride=1, padding=0, bias=False))

    #        self.getfilters = nn.Sequential(GetFilters(radius=2),
    #                                        nn.Conv2d(25, 75, kernel_size=1, stride=1, padding=0, bias=False))

    #        self.weight_lg1 = nn.Sequential(BasicConv(32, 32, kernel_size=3, padding=1),
    #                                        BasicConv(32, 32, relu=False, kernel_size=3, padding=1))
    #        self.weight_lg2 = nn.Sequential(BasicConv(32, 32, kernel_size=3, padding=1),
    #                                        BasicConv(32, 32, relu=False, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.conv0(x)
        temp = x

        x = self.conv1(x)
        rem = x
        sg1 = self.weight_sg1(x)
        sg1 = self.getweights(sg1)
        x = self.conv2(x) + rem
        rem = x
        sg2 = self.weight_sg2(x)
        sg2 = self.getweights(sg2)
        x = self.conv3(x) + rem
        sg3 = self.weight_sg3(x)
        sg3 = self.getweights(sg3)

        x = self.conv11(x)
        rem = x
        sg11 = self.weight_sg11(x)
        sg11 = self.getweights(sg11)
        x = self.conv12(x) + rem
        rem = x
        sg12 = self.weight_sg12(x)
        sg12 = self.getweights(sg12)
        x = self.conv13(x) + rem
        rem = x
        sg13 = self.weight_sg13(x)
        sg13 = self.getweights(sg13)
        x = self.conv14(x) + rem
        sg14 = self.weight_sg14(x)
        sg14 = self.getweights(sg14)

        #        lg1 = self.weight_lg1(temp)
        #        lg1 = self.getfilters(lg1)
        #        lg2 = self.weight_lg2(temp)
        #        lg2 = self.getfilters(lg2)

        return dict([
            ('sg1', sg1),
            ('sg2', sg2),
            ('sg3', sg3),
            ('sg11', sg11),
            ('sg12', sg12),
            ('sg13', sg13),
            ('sg14', sg14)])


#            ('lg1', lg1),
#            ('lg2', lg2)])

class Disp(nn.Module):

    def __init__(self, maxdisp=192, InChannel=32):
        super(Disp, self).__init__()
        self.maxdisp = maxdisp
        self.softmax = nn.Softmin(dim=1)
        self.disparity = DisparityRegression(maxdisp=self.maxdisp)
        #        self.conv32x1 = BasicConv(32, 1, kernel_size=3)
        if InChannel == 64:
            self.conv3d_2d = nn.Conv3d(InChannel, 1, (1, 1, 1), (1, 1, 1), (1, 1, 1), bias=False)
        else:
            self.conv3d_2d = nn.Conv3d(InChannel, 1, (3, 3, 3), (1, 1, 1), (1, 1, 1), bias=False)

    def forward(self, x):
        x = F.interpolate(self.conv3d_2d(x), [self.maxdisp + 1, x.size()[3] * 4, x.size()[4] * 4], mode='trilinear',
                          align_corners=False)
        x = torch.squeeze(x, 1)
        x = self.softmax(x)

        return self.disparity(x)


class SGABlock(nn.Module):
    def __init__(self, channels=32, refine=False):
        super(SGABlock, self).__init__()
        self.refine = refine
        if self.refine:
            self.bn_relu = nn.Sequential(nn.BatchNorm3d(channels),
                                         nn.ReLU(inplace=True))
            self.conv_refine = BasicConv(channels, channels, is_3d=True, kernel_size=3, padding=1, relu=False)
        #            self.conv_refine1 = BasicConv(8, 8, is_3d=True, kernel_size=1, padding=1)
        else:
            self.bn = nn.BatchNorm3d(channels)
        self.SGA = SGA()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        rem = x
        #        k1, k2, k3, k4 = torch.split(g, (x.size()[1]*5, x.size()[1]*5, x.size()[1]*5, x.size()[1]*5), 1)
        #        k1 = F.normalize(k1.view(x.size()[0], x.size()[1], 5, x.size()[3], x.size()[4]), p=1, dim=2)
        #        k2 = F.normalize(k2.view(x.size()[0], x.size()[1], 5, x.size()[3], x.size()[4]), p=1, dim=2)
        #        k3 = F.normalize(k3.view(x.size()[0], x.size()[1], 5, x.size()[3], x.size()[4]), p=1, dim=2)
        #        k4 = F.normalize(k4.view(x.size()[0], x.size()[1], 5, x.size()[3], x.size()[4]), p=1, dim=2)
        k1, k2, k3, k4 = torch.split(g, (5, 5, 5, 5), 1)
        k1 = F.normalize(k1, p=1, dim=1)
        k2 = F.normalize(k2, p=1, dim=1)
        k3 = F.normalize(k3, p=1, dim=1)
        k4 = F.normalize(k4, p=1, dim=1)
        x = self.SGA(x, k1, k2, k3, k4)
        if self.refine:
            x = self.bn_relu(x)
            x = self.conv_refine(x)
        else:
            x = self.bn(x)
        assert (x.size() == rem.size())
        x += rem
        return self.relu(x)
    #        return self.bn_relu(x)


class CostAggregation(nn.Module):
    def __init__(self, maxdisp=192):
        super(CostAggregation, self).__init__()
        self.maxdisp = maxdisp
        self.conv_start = BasicConv(64, 32, is_3d=True, kernel_size=3, padding=1, relu=False)
        #        self.conv0 = nn.Sequential(BasicConv(32, 32, is_3d=True, kernel_size=3, padding=1),
        #                                   BasicConv(32, 32, is_3d=True, kernel_size=3, padding=1))

        self.conv1a = BasicConv(32, 48, is_3d=True, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, is_3d=True, kernel_size=3, stride=2, padding=1)
        #        self.conv3a = BasicConv(64, 96, is_3d=True, kernel_size=3, stride=2, padding=1)

        self.deconv1a = Conv2x(48, 32, deconv=True, is_3d=True, relu=False)
        self.deconv2a = Conv2x(64, 48, deconv=True, is_3d=True)
        #        self.deconv3a = Conv2x(96, 64, deconv=True, is_3d=True)

        self.conv1b = Conv2x(32, 48, is_3d=True)
        self.conv2b = Conv2x(48, 64, is_3d=True)
        #        self.conv3b = Conv2x(64, 96, is_3d=True)

        self.deconv1b = Conv2x(48, 32, deconv=True, is_3d=True, relu=False)
        self.deconv2b = Conv2x(64, 48, deconv=True, is_3d=True)
        #        self.deconv3b = Conv2x(96, 64, deconv=True, is_3d=True)
        #        self.deconv0b = Conv2x(8, 8, deconv=True, is_3d=True)

        self.disp0 = Disp(self.maxdisp)
        self.disp1 = Disp(self.maxdisp)
        self.disp2 = Disp(self.maxdisp)
        self.sga1 = SGABlock(channels=32, refine=True)
        self.sga2 = SGABlock(channels=32, refine=True)
        self.sga3 = SGABlock(channels=32, refine=True)
        self.sga11 = SGABlock(channels=48, refine=True)
        self.sga12 = SGABlock(channels=48, refine=True)
        self.sga13 = SGABlock(channels=48, refine=True)
        self.sga14 = SGABlock(channels=48, refine=True)

    def forward(self, x, g):

        x = self.conv_start(x)
        #        rem0 = x
        #        x = self.conv0(x)
        x = self.sga1(x, g['sg1'])
        #        x = x + rem0
        rem0 = x

        if self.training:
            disp0 = self.disp0(x)

        x = self.conv1a(x)
        x = self.sga11(x, g['sg11'])
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        #        x = self.conv3a(x)
        #        rem3 = x

        #        x = self.deconv3a(x, rem2)
        #        rem2 = x
        x = self.deconv2a(x, rem1)
        x = self.sga12(x, g['sg12'])
        rem1 = x
        x = self.deconv1a(x, rem0)
        x = self.sga2(x, g['sg2'])
        rem0 = x
        if self.training:
            disp1 = self.disp1(x)

        x = self.conv1b(x, rem1)
        x = self.sga13(x, g['sg13'])
        rem1 = x
        x = self.conv2b(x, rem2)
        #        rem2 = x
        #        x = self.conv3b(x, rem3)

        #        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.sga14(x, g['sg14'])
        x = self.deconv1b(x, rem0)
        x = self.sga3(x, g['sg3'])

        #        disp2 = self.disp2(x, g['lg1'])
        disp2 = self.disp2(x)
        if self.training:
            return disp0, disp1, disp2
        else:
            return disp2


def freeze_layers(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False
    for param in model.parameters():
        param.requires_grad = False


class DSMNet(nn.Module):
    def __init__(self, maxdisp=192):
        super(DSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.conv_start = nn.Sequential(BasicConv(3, 16, kernel_size=3, padding=1),
                                        BasicConv(16, 32, kernel_size=3, padding=1))

        self.conv_x = BasicConv(32, 32, kernel_size=3, padding=1)
        self.conv_y = BasicConv(32, 32, kernel_size=3, padding=1)
        self.conv_refine = nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), bias=False)
        #        self.bn_relu =nn.Sequential(SELayer(channel=32, reduction=2),
        #                                    nn.ReLU(inplace=True))
        self.bn_relu = nn.Sequential(DomainNorm(32),
                                     nn.ReLU(inplace=True))
        #         self.bn_relu = nn.Sequential(nn.InstanceNorm2d(32, affine=True),
        #                                     nn.ReLU(inplace=True))
        self.feature = Feature()
        self.guidance = Guidance()
        self.cost_agg = CostAggregation(self.maxdisp)
        self.cv = GetCostVolume(int(self.maxdisp / 4))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            #            elif isinstance(m, (nn.InstanceNorm2d, BatchNorm3d)):
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            if isinstance(m, nn.BatchNorm3d):
                m.eval()
            if isinstance(m, nn.SyncBatchNorm):
                m.eval()

    def forward(self, x, y):
        g = self.conv_start(x)
        x = self.feature(x)
        #        x = self.testit(x)
        rem = x
        x = self.conv_x(x)

        y = self.feature(y)

        y = self.conv_y(y)
        x = self.cv(x, y)
        # print(x.shape)
        x1 = self.conv_refine(rem)
        x1 = F.interpolate(x1, [x1.size()[2] * 4, x1.size()[3] * 4], mode='bilinear', align_corners=False)
        x1 = self.bn_relu(x1)
        g = torch.cat((g, x1), 1)
        g = self.guidance(g)
        #        return rem, g
        return self.cost_agg(x, g)
