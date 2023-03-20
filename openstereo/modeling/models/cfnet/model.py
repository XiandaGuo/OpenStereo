import torch

from modeling.base_model import BaseModel
from .cfnet import CFNet as cfnet


class CFNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_network(self):
        self.net = cfnet(
            self.max_disp,
            self.model_cfg['replace_mish'],
        )

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
                    "disp_est": res[0][0]
                },
                "visual_summary": {
                    'image/test/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                    'image/test/disp_c': res[0][0][0],
                }
            }
            if 'disp_gt' in inputs:
                output['visual_summary'] = {
                    'image/val/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                    'image/val/disp_c': torch.cat([inputs['disp_gt'][0], res[0][0][0]], dim=0),
                }
        return output
