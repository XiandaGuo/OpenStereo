import torch.nn as nn
import torch.nn.functional as F
from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d
from stereo.modeling.cost_volume.cost_volume import correlation_volume
from stereo.modeling.disp_pred.disp_regression import disparity_regression
from stereo.modeling.disp_refinement.disp_refinement import context_upsample

from .backbone import Backbone, FPNLayer
from .aggregation import Aggregation


class LightStereo(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.max_disp = cfgs.MAX_DISP
        self.left_att = cfgs.LEFT_ATT

        # backbobe
        self.backbone = Backbone(cfgs.get('BACKCONE', 'MobileNetv2'))

        # aggregation
        self.cost_agg = Aggregation(in_channels=48,
                                    left_att=self.left_att,
                                    blocks=cfgs.AGGREGATION_BLOCKS,
                                    expanse_ratio=cfgs.EXPANSE_RATIO,
                                    backbone_channels=self.backbone.output_channels)

        # disp refine
        self.refine_1 = nn.Sequential(
            BasicConv2d(self.backbone.output_channels[0], 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(24, 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.ReLU))

        self.stem_2 = nn.Sequential(
            BasicConv2d(3, 16, kernel_size=3, stride=2, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(16, 16, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU))
        self.refine_2 = FPNLayer(24, 16)

        self.refine_3 = BasicDeconv2d(16, 9, kernel_size=4, stride=2, padding=1)

    def forward(self, data):
        image1 = data['left']
        image2 = data['right']

        features_left = self.backbone(image1)
        features_right = self.backbone(image2)

        gwc_volume = correlation_volume(features_left[0], features_right[0], self.max_disp // 4)
        encoding_volume = self.cost_agg(gwc_volume, features_left)  # [bz, 1, max_disp/4, H/4, W/4]

        prob = F.softmax(encoding_volume[0].squeeze(1), dim=1)  # [bz, max_disp/4, H/4, W/4]
        init_disp = disparity_regression(prob, self.max_disp // 4)  # [bz, 1, H/4, W/4]

        xspx = self.refine_1(features_left[0])
        xspx = self.refine_2(xspx, self.stem_2(image1))
        xspx = self.refine_3(xspx)
        spx_pred = F.softmax(xspx, 1)  # [bz, 9, H, W]
        disp_pred = context_upsample(init_disp * 4., spx_pred.float()).unsqueeze(1)  # # [bz, 1, H, W]

        result = {'disp_pred': disp_pred}

        if self.training:
            disp_4 = F.interpolate(init_disp, image1.shape[2:], mode='bilinear', align_corners=False)
            disp_4 *= 4
            result['disp_4'] = disp_4

        return result

    def get_loss(self, model_pred, input_data):
        disp_gt = input_data["disp"]  # [bz, h, w]
        disp_gt = disp_gt.unsqueeze(1)  # [bz, 1, h, w]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, 1, h, w]

        disp_pred = model_pred['disp_pred']
        loss = 1.0 * F.smooth_l1_loss(disp_pred[mask], disp_gt[mask], reduction='mean')

        disp_4 = model_pred['disp_4']
        loss += 0.3 * F.smooth_l1_loss(disp_4[mask], disp_gt[mask], reduction='mean')

        loss_info = {'scalar/train/loss_disp': loss.item()}

        return loss, loss_info
