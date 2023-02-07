from modeling.base_model import BaseModel
from utils import Odict
from .DSMNet import DSMNet as Net1x
from .DSMNet2x2 import DSMNet as Net2x

from modeling.losses import Weighted_Smooth_l1_Loss


class GALoss:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_func = Weighted_Smooth_l1_Loss(weights=[0.2, 0.6, 1])

    def __call__(self, training_output, disp_gt, mask):
        training_disp = training_output['training_disp']
        loss_info = Odict()
        pred_disp = training_disp['disp_est']
        loss = self.loss_func(pred_disp, disp_gt, mask)
        loss_info['scalar/GALoss'] = loss
        return loss, loss_info


class DSMNet1x(BaseModel):
    def __init__(self, *args, **kwargs):
        self.maxdisp = 192
        super().__init__(*args, **kwargs)

    def build_network(self, model_cfg):
        """Build the network."""
        self.net = Net1x(
            maxdisp=self.maxdisp,
        )

    def get_loss_func(self, loss_cfg):
        """Build the loss."""
        return GALoss()

    def forward(self, inputs):
        """Forward the network."""
        ref_img = inputs["ref_img"]
        tgt_img = inputs["tgt_img"]
        res = self.net(ref_img, tgt_img)

        if self.training:
            output = {
                "training_disp": {
                    "disp_est": res
                },
                "visual_summary": {}
            }
        else:
            output = {
                "inference_disp": {
                    "disp_est": res
                },
                "visual_summary": {}
            }

        return output


class DSMNet2x(DSMNet1x):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_network(self, model_cfg):
        """Build the network."""
        self.net = Net2x(
            maxdisp=self.maxdisp,
        )
