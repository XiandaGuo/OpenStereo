import torch
import torch.nn.functional as F

from libs.GANet.modules.GANet import MyLoss2
from modeling.base_model import BaseModel
from utils import Odict
from .GANet_deep import GANet as GANet_deep


class GaLoss:
    def __init__(self, is_kitti=False):
        super(GaLoss, self).__init__()
        self.is_kitti = is_kitti
        self.criterion = MyLoss2(thresh=3, alpha=2)

    def __call__(self, training_output):
        training_disp = training_output['disp']
        pred_disp = training_disp['disp_ests']
        disp_gt = training_disp['disp_gt']
        mask = training_disp['mask']
        disp0, disp1, disp2 = pred_disp
        if self.is_kitti:
            loss = 0.2 * F.smooth_l1_loss(disp0[mask], disp_gt[mask], reduction='mean') \
                   + 0.6 * F.smooth_l1_loss(disp1[mask], disp_gt[mask], reduction='mean') \
                   + self.criterion(disp2[mask], disp_gt[mask])
        else:
            loss = 0.2 * F.smooth_l1_loss(disp0[mask], disp_gt[mask], reduction='mean') \
                   + 0.6 * F.smooth_l1_loss(disp1[mask], disp_gt[mask], reduction='mean') \
                   + F.smooth_l1_loss(disp2[mask], disp_gt[mask], reduction='mean')

        loss_info = Odict()
        loss_info['scalar/train/loss_disp'] = loss
        loss_info['scalar/train/loss_sum'] = loss
        return loss, loss_info


class GANet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_loss_fn(self):
        """Build the loss."""
        loss_cfg = self.cfg['loss_cfg']
        is_kitti = loss_cfg.get("is_kitti", False)
        self.loss_fn = GaLoss(is_kitti)

    def build_network(self):
        """Build the network."""
        self.net = GANet_deep(maxdisp=self.max_disp)

    def init_parameters(self):
        return

    def forward(self, inputs):
        """Forward the network."""
        ref_img = inputs["ref_img"]
        tgt_img = inputs["tgt_img"]
        res = self.net(ref_img, tgt_img)

        if self.training:
            [disp0, disp1, disp2] = res
            output = {
                "training_disp": {
                    "disp": {
                        "disp_ests": [disp0, disp1, disp2],
                        "disp_gt": inputs['disp_gt'],
                        "mask": inputs['mask']
                    },
                },
                "visual_summary": {
                    'image/train/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                    'image/train/disp_c': torch.cat([inputs['disp_gt'][0], res[-1][0]], dim=0),
                },
            }
        else:
            output = {
                "inference_disp": {
                    "disp_est": res
                },
                "visual_summary": {
                    'image/test/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                    'image/test/disp_c': res[0],
                }
            }
            if 'disp_gt' in inputs:
                output['visual_summary'] = {
                    'image/val/disp_c': torch.cat([inputs['disp_gt'][0], res[0]], dim=0),
                    'image/val/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                }

        return output
