import torch
import torch.nn.functional as F

from modeling.base_model import BaseModel
from .stackhourglass import PSMNet
from . import loss_functions as lf


class LacGwcLoss:
    def __init__(self, max_disp=192):
        self.max_disp = max_disp
        self.info = {}

    def loss_fn(self, training_output):
        training_disp = training_output['disp']
        pred_disp = training_disp['disp_ests']
        disp_gt = training_disp['disp_gt']
        mask = training_disp['mask']
        pred1, pred2, pred3, distribute1, distribute2, distribute3, distributer, predr = pred_disp

        loss1 = 0.5 * F.smooth_l1_loss(pred1[mask], disp_gt[mask]) + \
                0.7 * F.smooth_l1_loss(pred2[mask], disp_gt[mask]) + \
                F.smooth_l1_loss(pred3[mask], disp_gt[mask])

        gt_distribute = lf.disp2distribute(disp_gt, self.max_disp, b=2)
        loss2 = 0.5 * lf.CEloss(disp_gt, self.max_disp, gt_distribute, distribute1) + \
                0.7 * lf.CEloss(disp_gt, self.max_disp, gt_distribute, distribute2) + \
                lf.CEloss(disp_gt, self.max_disp, gt_distribute, distribute3)
        loss1 += F.smooth_l1_loss(predr[mask], disp_gt[mask])
        loss2 += lf.CEloss(disp_gt, self.max_disp, gt_distribute, distributer)

        return 0.1 * loss1 + loss2

    def __call__(self, training_output):
        loss = self.loss_fn(training_output)
        self.info['scalar/disp_loss'] = loss
        self.info['scalar/loss_sum'] = loss
        return loss, self.info


class LacGwcNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_network(self, model_cfg):
        self.net = PSMNet(model_cfg['base_config']['max_disp'])

    def get_loss_func(self, loss_cfg):
        return LacGwcLoss(max_disp=192)

    def forward(self, inputs):
        """Forward the network."""
        ref_img = inputs["ref_img"]
        tgt_img = inputs["tgt_img"]
        if self.training:
            res = self.net(ref_img, tgt_img)
            output = {
                "training_disp": {
                    "disp": {
                        "disp_ests": res,
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
            res = self.net(ref_img, tgt_img)
            output = {
                "inference_disp": {
                    "disp_est": res
                },
                "visual_summary": {
                    'image/val/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                    'image/val/disp_c': torch.cat([inputs['disp_gt'][0], res[0]], dim=0),
                }
            }
        return output
