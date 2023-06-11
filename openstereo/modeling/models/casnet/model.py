import torch
import torch.nn.functional as F

from modeling.base_model import BaseModel
from utils import Odict
from .cas_gwc import GwcNet as CasGwcNet
from .cas_psm import PSMNet as CasPSMNet


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
        loss = self.stereo_loss(pred_disp, disp_gt, mask, self.dlossw)
        loss_info['scalar/train/loss_disp'] = loss
        loss_info['scalar/train/loss_sum'] = loss
        return loss, loss_info

    @staticmethod
    def stereo_loss(inputs, target, mask, dlossw=None):
        total_loss = torch.tensor(0.0, dtype=target.dtype, device=target.device, requires_grad=False)
        for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
            disp0, disp1, disp2, disp3 = stage_inputs["pred0"], stage_inputs["pred1"], stage_inputs["pred2"], \
                stage_inputs[
                    "pred3"]

            loss = 0.5 * F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') + \
                   0.5 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') + \
                   0.7 * F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean') + \
                   1.0 * F.smooth_l1_loss(disp3[mask], target[mask], reduction='mean')

            if dlossw is not None:
                stage_idx = int(stage_key.replace("stage", "")) - 1
                total_loss += dlossw[stage_idx] * loss
            else:
                total_loss += loss

        return total_loss


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

    def build_network(self):
        """Build the network."""
        model_type = self.model_cfg['model_type']
        if model_type == 'psmnet':
            self.net = CasPSMNet(
                maxdisp=self.maxdisp,
                ndisps=self.ndisps,
                disp_interval_pixel=self.disp_interval_pixel,
                using_ns=self.using_ns,
                ns_size=self.ns_size,
                grad_method=self.grad_method,
                cr_base_chs=self.cr_base_chs
            )
        elif model_type == 'gwcnet':
            self.net = CasGwcNet(
                maxdisp=self.maxdisp,
                ndisps=self.ndisps,
                disp_interval_pixel=self.disp_interval_pixel,
                using_ns=self.using_ns,
                ns_size=self.ns_size,
                grad_method=self.grad_method,
                cr_base_chs=self.cr_base_chs
            )

    def build_loss_fn(self):
        """Build the loss."""
        self.loss_fn = CasStereoLoss(dlossw=self.dlossw)

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
