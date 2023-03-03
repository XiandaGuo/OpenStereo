import torch

from modeling.base_model import BaseModel
from utils import Odict
from .loss import stereo_psmnet_loss
from .psmnet import PSMNet


class CasStereoLoss:
    def __init__(self, dlossw):
        super().__init__()
        self.dlossw = dlossw

    def __call__(self, training_output):
        training_disp = training_output['disp']
        pred_disp = training_disp['disp_ests']
        disp_gt = training_disp['disp_gt']
        mask = training_disp['mask']
        loss_info = Odict()
        loss = stereo_psmnet_loss(pred_disp, disp_gt, mask, self.dlossw)
        loss_info['scalar/disp_loss'] = loss
        loss_info['scalar/loss_sum'] = loss
        return loss, loss_info


class CasStereoNet(BaseModel):
    def __init__(self, *args, **kwargs):
        self.maxdisp = 192
        self.ndisps = [48, 24]
        self.disp_interval_pixel = [4, 1]
        self.dlossw = [0.5, 1.0, 2.0]
        self.using_ns = True
        self.ns_size = 3
        self.grad_method = 'detach'
        self.cr_base_chs = [32, 32, 16]
        super().__init__(*args, **kwargs)

    def init_parameters(self):
        return

    def build_network(self, model_cfg):
        """Build the network."""
        self.net = PSMNet(
            maxdisp=self.maxdisp,
            ndisps=self.ndisps,
            disp_interval_pixel=self.disp_interval_pixel,
            using_ns=self.using_ns,
            ns_size=self.ns_size,
            grad_method=self.grad_method,
            cr_base_chs=self.cr_base_chs
        )
        # self.net = GwcNet(
        #     maxdisp=self.maxdisp,
        #     ndisps=self.ndisps,
        #     disp_interval_pixel=self.disp_interval_pixel,
        #     using_ns=self.using_ns,
        #     ns_size=self.ns_size,
        #     grad_method=self.grad_method,
        #     cr_base_chs=self.cr_base_chs
        # )

    def get_loss_func(self, loss_cfg):
        """Build the loss."""
        return CasStereoLoss(dlossw=self.dlossw)

    def forward(self, inputs):
        """Forward the network."""
        ref_img = inputs["ref_img"]
        tgt_img = inputs["tgt_img"]
        res = self.net(ref_img, tgt_img)

        if self.training:
            output = {
                "training_disp": {
                    "disp": {
                        "disp_ests": res,
                        "disp_gt": inputs["disp_gt"],
                        "mask": inputs["mask"]
                    }
                },
                "visual_summary": {
                    "image/train/image_c": torch.cat([ref_img[0], tgt_img[0]], dim=1),
                    "image/train/disp_c": torch.cat(
                        [inputs["disp_gt"][0], res[f"stage{len(self.ndisps)}"]["pred"][0]],
                        dim=0
                    )
                }
            }
        else:
            disp_est = res[f"stage{len(self.ndisps)}"]['pred']
            output = {
                "inference_disp": {
                    "disp_est": disp_est
                },
                "visual_summary": {
                    "image/test/image_c": torch.cat([ref_img[0], tgt_img[0]], dim=1),
                    "image/test/disp_c": disp_est[0]
                }
            }
            if 'disp_gt' in inputs:
                disp_gt = inputs['disp_gt']
                output['visual_summary'] = {
                    "image/val/image_c": torch.cat([ref_img[0], tgt_img[0]], dim=1),
                    "image/val/disp_c": torch.cat([disp_gt[0], disp_est[0]], dim=0)
                }

        return output
