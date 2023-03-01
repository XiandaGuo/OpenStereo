from .CoEx import CoEx

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.base_model import BaseModel


class CoExNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_network(self, model_cfg):
        self.net = CoEx()

    def init_parameters(self):
        pass

    def forward(self, inputs):
        """Forward the network."""
        ref_img = inputs["ref_img"]
        tgt_img = inputs["tgt_img"]
        shape = ref_img.shape
        res = self.net(ref_img, tgt_img)
        if self.training:
            r0 = res[0]
            r1 = res[1]
            r1 = F.interpolate(r1.unsqueeze(1), size=(shape[2], shape[3]), mode='bilinear').squeeze(1)
            disp_pred = [r0, r1]
            output = {
                "training_disp": {
                    "disp": {
                        "disp_ests": disp_pred,
                        "disp_gt": inputs['disp_gt'],
                        "mask": inputs['mask']
                    },
                },
                "visual_summary": {
                    'image/train/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                    'image/train/disp_c': torch.cat([inputs['disp_gt'][0], disp_pred[0][0]], dim=0),
                },
            }

        else:
            disp_pred = res[0]
            output = {
                "inference_disp": {
                    "disp_est": disp_pred,
                },
                "visual_summary": {
                    'image/test/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                    'image/test/disp_c': disp_pred[0],
                }
            }
            if 'disp_gt' in inputs:
                output['visual_summary'] = {
                    'image/val/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                    'image/val/disp_c': torch.cat([inputs['disp_gt'][0], disp_pred[0]], dim=0),
                }
        return output
