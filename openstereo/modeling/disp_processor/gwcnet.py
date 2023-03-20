import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.common.hourglass import Hourglass


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            bias=False
        ),
        nn.BatchNorm3d(out_channels)
    )


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)


class GwcDispProcessor(nn.Module):
    def __init__(self, maxdisp=192, downsample=4, num_groups=40, use_concat_volume=True, concat_channels=12, *args,
                 **kwargs):
        super().__init__()

        self.maxdisp = maxdisp
        self.downsample = downsample
        self.num_groups = num_groups
        self.use_concat_volume = use_concat_volume
        self.concat_channels = concat_channels if use_concat_volume else 0

        self.dres0 = nn.Sequential(
            convbn_3d(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.dres1 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(32, 32, 3, 1, 1)
        )

        self.dres2 = Hourglass(32)

        self.dres3 = Hourglass(32)

        self.dres4 = Hourglass(32)

        self.classif0 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)
        )

        self.classif1 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)
        )

        self.classif2 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)
        )

        self.classif3 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)
        )

    def forward(self, inputs):
        volume = inputs['cost_volume']
        h, w = inputs['ref_img'].shape[2:]
        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0

        out1 = self.dres2(cost0)
        out2 = self.dres3(out1)
        out3 = self.dres4(out2)

        if self.training:
            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)
            cost3 = self.classif3(out3)

            cost0 = F.interpolate(cost0, [self.maxdisp, h, w], mode='trilinear', align_corners=False)
            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0, self.maxdisp)

            cost1 = F.interpolate(cost1, [self.maxdisp, h, w], mode='trilinear', align_corners=False)
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, self.maxdisp)

            cost2 = F.interpolate(cost2, [self.maxdisp, h, w], mode='trilinear', align_corners=False)
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)

            cost3 = F.interpolate(cost3, [self.maxdisp, h, w], mode='trilinear', align_corners=False)
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)

            output = {
                "training_disp": {
                    "disp": {
                        "disp_ests": [pred0,pred1,pred2,pred3],
                        "disp_gt": inputs['disp_gt'],
                        "mask": inputs['mask']
                    },
                },
                "visual_summary": {
                    'image/train/image_c': torch.cat([inputs['ref_img'][0], inputs['tgt_img'][0]], dim=1),
                    'image/train/disp_c': torch.cat([inputs['disp_gt'][0], pred3[0]], dim=0),
                }
            }
            return output

        else:
            cost3 = self.classif3(out3)
            cost3 = F.interpolate(cost3, [self.maxdisp, h, w], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)

            output = {
                "inference_disp": {
                    "disp_est": pred3,
                },
                "visual_summary": {
                    'image/test/image_c': torch.cat([inputs['ref_img'][0], inputs['tgt_img'][0]], dim=1),
                    'image/test/disp_c': pred3[0]
                }
            }
            if 'disp_gt' in inputs:
                output['visual_summary']={
                    'image/val/image_c': torch.cat([inputs['ref_img'][0], inputs['tgt_img'][0]], dim=1),
                    'image/val/disp_c': torch.cat([inputs['disp_gt'][0], pred3[0]], dim=0),
                }
            return output

    def input_output(self):
        return {
            "inputs": ["cost_volume", "disp_shape"],
            "outputs": ["training_disp", "inference_disp", "visual_summary"]
        }
