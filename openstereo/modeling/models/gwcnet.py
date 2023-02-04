from modeling.base_model import BaseModel
from modeling.losses import Weighted_Smooth_l1_Loss
from utils import Odict


class GwcLoss:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_func = Weighted_Smooth_l1_Loss()

    def __call__(self, training_output, disp_gt, mask):
        training_disp = training_output['training_disp']
        loss_info = Odict()
        pred_disp = training_disp['disp_est']
        middle = training_disp['disp_hidden']
        loss = self.loss_func(middle + [pred_disp], disp_gt, mask)
        loss_info['scalar/Weighted_Smooth_l1_Loss'] = loss
        return loss, loss_info


class GwcNet(BaseModel):
    def __int__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, inputs):
        """Forward the network."""
        backbone_out = self.Backbone(inputs)
        cost_out = self.CostProcessor(backbone_out)
        # GwcNet disparity processor need the disparity map size
        cost_out.update({'disp_shape': inputs['ref_img'].shape[-2:]})
        disp_out = self.DispProcessor(cost_out)
        return disp_out

    def get_loss_func(self, loss_cfg):
        return GwcLoss()
