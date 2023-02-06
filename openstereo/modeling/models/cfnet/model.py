from modeling.base_model import BaseModel
from utils import Odict
from .cfnet import CFNet as cfnet
from .loss import model_loss


class CFLoss:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_func = model_loss

    def __call__(self, training_output, disp_gt, mask):
        training_disp = training_output['training_disp']
        loss_info = Odict()
        pred_disp = training_disp['disp_est']
        loss = self.loss_func(pred_disp, disp_gt, mask)
        loss_info['scalar/CFLoss'] = loss
        return loss, loss_info


class CFNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_network(self, model_cfg):
        self.net = cfnet(model_cfg['base_config']['max_disp'])

    def forward(self, inputs):
        """Forward the network."""
        ref_img = inputs["ref_img"]
        tgt_img = inputs["tgt_img"]
        if self.training:
            res = self.net(ref_img, tgt_img)
            output = {
                "training_disp": {
                    "disp_est": res
                },
                "visual_summary": {}
            }
        else:
            res = self.net(ref_img, tgt_img)
            output = {
                "inference_disp": {
                    "disp_est": res[0][0]
                },
                "visual_summary": {}
            }
        return output

    def get_loss_func(self, loss_cfg):
        return CFLoss()
