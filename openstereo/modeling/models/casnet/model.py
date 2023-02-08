from functools import partial

from modeling.base_model import BaseModel
from utils import Odict
from .psmnet import PSMNet
from .loss import stereo_psmnet_loss


class CasStereoLoss:
    def __init__(self, dlossw, **kwargs):
        super().__init__(**kwargs)
        self.loss_func = partial(stereo_psmnet_loss, dlossw=dlossw)

    def __call__(self, training_output):
        training_disp = training_output['disp']
        pred_disp = training_disp['disp_ests']
        disp_gt = training_disp['disp_gt']
        mask = training_disp['mask']
        loss_info = Odict()
        loss = self.loss_func(pred_disp, disp_gt, mask)
        loss_info['scalar/CasStereoLoss'] = loss
        return loss, loss_info


class CasStereoNet(BaseModel):
    def __init__(self, *args, **kwargs):
        self.maxdisp = 192
        self.ndisps = [48, 24]
        self.disp_interval_pixel = [4, 1]
        self.using_ns = True
        self.ns_size = 3
        self.grad_method = 'detach'
        self.cr_base_chs = [32, 32, 16]
        super().__init__(*args, **kwargs)

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

    def get_loss_func(self, loss_cfg):
        """Build the loss."""
        return CasStereoLoss(dlossw=[0.5, 1.0, 2.0])

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
                "visual_summary": {}
            }
        else:
            output = {
                "inference_disp": {
                    "disp_est": res[f"stage{(len(self.ndisps) - 1)}"]['pred']
                },
                "visual_summary": {}
            }

        return output
