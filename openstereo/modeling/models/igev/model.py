import torch
import torch.nn.functional as F

from modeling.base_model import BaseModel
from utils import Odict
from types import SimpleNamespace

from .igev_stereo import IGEVStereo


class IGEVLoss:
    def __init__(self, loss_gamma=0.9, max_disp=192):
        super().__init__()
        self.loss_gamma = loss_gamma
        self.max_disp = max_disp

    def __call__(self, training_output):
        training_disp = training_output['disp']
        pred_disp = training_disp['disp_ests']
        disp_gt = training_disp['disp_gt']
        mask = training_disp['mask']

        init_disp, disp_preds = pred_disp

        loss = self.sequence_loss(disp_preds, init_disp, disp_gt, mask.float())

        loss_info = Odict()
        loss_info['scalar/train/loss_disp'] = loss.item()
        loss_info['scalar/train/loss_sum'] = loss.item()
        return loss, loss_info

    @staticmethod
    def sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, loss_gamma=0.9, max_disp=192):
        """ Loss function defined over sequence of flow predictions """

        n_predictions = len(disp_preds)
        assert n_predictions >= 1
        disp_loss = 0.0
        disp_gt = disp_gt.unsqueeze(1)
        mag = torch.sum(disp_gt ** 2, dim=1).sqrt()
        valid = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
        assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
        assert not torch.isinf(disp_gt[valid.bool()]).any()

        disp_loss += 1.0 * F.smooth_l1_loss(disp_init_pred[valid.bool()], disp_gt[valid.bool()], size_average=True)
        for i in range(n_predictions):
            adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
            i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
            i_loss = (disp_preds[i] - disp_gt).abs()
            assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
            disp_loss += i_weight * i_loss[valid.bool()].mean()
        return disp_loss


class IGEV(BaseModel):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)


    def build_network(self):
        model_cfg = self.model_cfg
        self.net = IGEVStereo(model_cfg['base_config'])

    def init_parameters(self):
        return

    def build_loss_fn(self):
        """Build the loss."""
        self.loss_fn = IGEVLoss()

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
                        "disp_gt": inputs['disp_gt'],
                        "mask": inputs['mask']
                    },
                },
                "visual_summary": {
                    'image/train/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                    'image/train/disp_c': torch.cat([inputs['disp_gt'][0], res[0].squeeze(1)[0]], dim=0),
                },
            }
        else:
            disp_pred = res.squeeze(1)
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
