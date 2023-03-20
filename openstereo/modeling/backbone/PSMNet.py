import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.common.basic_layers import conv_bn, conv_bn_relu, BasicBlock


class PSMNet(nn.Module):
    """
    Backbone proposed in PSMNet.
    Args:
        in_planes (int): the channels of input
        batch_norm (bool): whether use batch normalization layer, default True
    Inputs:
        l_img (Tensor): left image, in [BatchSize, 3, Height, Width] layout
        r_img (Tensor): right image, in [BatchSize, 3, Height, Width] layout
    Outputs:
        l_fms (Tensor): left image feature maps, in [BatchSize, 32, Height//4, Width//4] layout

        r_fms (Tensor): right image feature maps, in [BatchSize, 32, Height//4, Width//4] layout
    """

    def __init__(self, in_planes=3, batch_norm=True):
        super().__init__()
        self.in_planes = in_planes
        self.batch_norm = batch_norm

        self.firstconv = nn.Sequential(
            conv_bn_relu(batch_norm, self.in_planes, 32, 3, 2, 1, 1, bias=False),
            conv_bn_relu(batch_norm, 32, 32, 3, 1, 1, 1, bias=False),
            conv_bn_relu(batch_norm, 32, 32, 3, 1, 1, 1, bias=False),
        )

        # For building Basic Block
        self.in_planes = 32

        self.layer1 = self._make_layer(batch_norm, BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(batch_norm, BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(batch_norm, BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(batch_norm, BasicBlock, 128, 3, 1, 2, 2)

        self.branch1 = nn.Sequential(
            nn.AvgPool2d((64, 64), stride=(64, 64)),
            conv_bn_relu(batch_norm, 128, 32, 1, 1, 0, 1, bias=False),
        )
        self.branch2 = nn.Sequential(
            nn.AvgPool2d((32, 32), stride=(32, 32)),
            conv_bn_relu(batch_norm, 128, 32, 1, 1, 0, 1, bias=False),
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d((16, 16), stride=(16, 16)),
            conv_bn_relu(batch_norm, 128, 32, 1, 1, 0, 1, bias=False),
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d((8, 8), stride=(8, 8)),
            conv_bn_relu(batch_norm, 128, 32, 1, 1, 0, 1, bias=False),
        )
        self.lastconv = nn.Sequential(
            conv_bn_relu(batch_norm, 320, 128, 3, 1, 1, 1, bias=False),
            nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, dilation=1, bias=False)
        )

    def _make_layer(self, batch_norm, block, out_planes, blocks, stride, padding, dilation):
        downsample = None
        if stride != 1 or self.in_planes != out_planes * block.expansion:
            downsample = conv_bn(
                batch_norm, self.in_planes, out_planes * block.expansion,
                kernel_size=1, stride=stride, padding=0, dilation=1
            )

        layers = []
        layers.append(
            block(batch_norm, self.in_planes, out_planes, stride, downsample, padding, dilation)
        )
        self.in_planes = out_planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(batch_norm, self.in_planes, out_planes, 1, None, padding, dilation)
            )

        return nn.Sequential(*layers)

    def _forward(self, x):
        output_2_0 = self.firstconv(x)
        output_2_1 = self.layer1(output_2_0)
        output_4_0 = self.layer2(output_2_1)
        output_4_1 = self.layer3(output_4_0)
        output_8 = self.layer4(output_4_1)

        output_branch1 = self.branch1(output_8)
        output_branch1 = F.interpolate(
            output_branch1, (output_8.size()[2], output_8.size()[3]),
            mode='bilinear', align_corners=True
        )

        output_branch2 = self.branch2(output_8)
        output_branch2 = F.interpolate(
            output_branch2, (output_8.size()[2], output_8.size()[3]),
            mode='bilinear', align_corners=True
        )

        output_branch3 = self.branch3(output_8)
        output_branch3 = F.interpolate(
            output_branch3, (output_8.size()[2], output_8.size()[3]),
            mode='bilinear', align_corners=True
        )

        output_branch4 = self.branch4(output_8)
        output_branch4 = F.interpolate(
            output_branch4, (output_8.size()[2], output_8.size()[3]),
            mode='bilinear', align_corners=True
        )

        output_feature = torch.cat(
            (output_4_0, output_8, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature

    def forward(self, inputs):
        ref_img = inputs["ref_img"]
        tgt_img = inputs["tgt_img"]
        l_fms = self._forward(ref_img)
        r_fms = self._forward(tgt_img)
        return {
            "ref_feature": l_fms,
            "tgt_feature": r_fms
        }

    def input_output(self):
        return {
            "inputs": ["ref_img", "tgt_img"],
            "outputs": ["ref_feature", "tgt_feature"]
        }
