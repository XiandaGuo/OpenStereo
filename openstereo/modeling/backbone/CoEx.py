import math

import timm
import torch
import torch.nn as nn

from modeling.common.model_basics.CoEx import BasicConv, Conv2x


class FeatUp(nn.Module):
    def __init__(self):
        super().__init__()
        # self.cfg = cfg['backbone']
        chans = [16, 24, 32, 96, 160]
        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)
        self.deconv16_8 = Conv2x(chans[3] * 2, chans[2], deconv=True, concat=True)
        self.deconv8_4 = Conv2x(chans[2] * 2, chans[1], deconv=True, concat=True)
        self.conv4 = BasicConv(chans[1] * 2, chans[1] * 2, kernel_size=3, stride=1, padding=1)
        self.weight_init()

    def forward(self, featL):
        x4, x8, x16, x32 = featL
        x16 = self.deconv32_16(x32, x16)
        x8 = self.deconv16_8(x16, x8)
        x4 = self.deconv8_4(x8, x4)
        x4 = self.conv4(x4)
        return [x4, x8, x16, x32]

    def weight_init(self):
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


class Feature(nn.Module):
    def __init__(self):
        super().__init__()
        self.type = 'mobilenetv2_100'
        layers = [1, 2, 3, 5, 6]
        self.pre_trained = True
        model = timm.create_model(self.type, pretrained=self.pre_trained, features_only=True)
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])
        self.up = FeatUp()

    def forward(self, x):
        x = self.bn1(self.conv_stem(x))
        x2 = self.block0(x)
        x4 = self.block1(x2)
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)
        return [x4, x8, x16, x32]


class CoExBackbone(nn.Module):
    def __init__(self, spixel_branch_channels=[32, 48]):
        super().__init__()
        self.feat = Feature()
        self.up = FeatUp()
        self.spixel_branch_channels = spixel_branch_channels
        self.stem_2 = nn.Sequential(
            BasicConv(3, self.spixel_branch_channels[0], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(self.spixel_branch_channels[0], self.spixel_branch_channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.spixel_branch_channels[0]),
            nn.ReLU()
        )
        self.stem_4 = nn.Sequential(
            BasicConv(self.spixel_branch_channels[0], self.spixel_branch_channels[1], kernel_size=3,
                      stride=2, padding=1),
            nn.Conv2d(self.spixel_branch_channels[1], self.spixel_branch_channels[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.spixel_branch_channels[1]),
            nn.ReLU()
        )

    def forward(self, inputs):
        ref_img = inputs["ref_img"]
        tgt_img = inputs["tgt_img"]
        ref_feature = self.feat(ref_img)
        ref_feature = self.up(ref_feature)
        tgt_feature = self.feat(tgt_img)
        tgt_feature = self.up(tgt_feature)

        stem_2x = self.stem_2(ref_img)
        stem_4x = self.stem_4(stem_2x)

        stem_2y = self.stem_2(tgt_img)
        stem_4y = self.stem_4(stem_2y)

        ref_feature[0] = torch.cat([ref_feature[0], stem_4x], dim=1)
        tgt_feature[0] = torch.cat([tgt_feature[0], stem_4y], dim=1)

        return {
            "ref_feature": ref_feature,
            "tgt_feature": tgt_feature,
            "stem_2x": stem_2x,
            # "stem_2y": stem_2y
        }
