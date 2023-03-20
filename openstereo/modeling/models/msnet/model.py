import torch

from modeling.base_model import BaseModel
from .MSNet2D import MSNet2D
from .MSNet3D import MSNet3D


class MSNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_network(self):
        if self.model_cfg['model_type'] == '3D':
            self.net = MSNet3D(self.max_disp)
        elif self.model_cfg['model_type'] == '2D':
            self.net = MSNet2D(self.max_disp)
        else:
            raise NotImplementedError

    def init_parameters(self):
        return

    def forward(self, inputs):
        """Forward the network."""
        ref_img = inputs["ref_img"]
        tgt_img = inputs["tgt_img"]
        res = self.net(ref_img, tgt_img)
        if self.training:
            pred0, pred1, pred2, pred3 = res
            output = {
                "training_disp": {
                    "disp": {
                        "disp_ests": [pred0, pred1, pred2, pred3],
                        "disp_gt": inputs['disp_gt'],
                        "mask": inputs['mask']
                    },
                },
                "visual_summary": {
                    'image/train/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                    'image/train/disp_c': torch.cat([inputs['disp_gt'][0], pred3[0]], dim=0),
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
