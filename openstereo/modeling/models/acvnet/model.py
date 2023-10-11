import torch
import torch.nn.functional as F

from modeling.base_model import BaseModel
from .acvnet import ACVNet as Net
from .acvnet_small import ACVNetSmall as NetSmall


class ACVLoss:
    def __init__(self, loss_type='attn_only'):
        super().__init__()
        assert loss_type in ['attn_only', 'freeze_attn', 'full', 'test'], f"loss_type {loss_type} not supported"
        self.loss_fn = None
        if loss_type == 'attn_only':
            self.loss_fn = self.model_loss_train_attn_only
        elif loss_type == 'freeze_attn':
            self.loss_fn = self.model_loss_train_freeze_attn
        elif loss_type == 'full':
            self.loss_fn = self.model_loss_train
        elif loss_type == 'test':
            self.loss_fn = self.model_loss_test
        else:
            raise NotImplementedError(f"loss_type {loss_type} not supported")

    @staticmethod
    def model_loss_train_attn_only(disp_ests, disp_gt, mask):
        weights = [1.0]
        all_losses = []
        for disp_est, weight in zip(disp_ests, weights):
            all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
        return sum(all_losses)

    @staticmethod
    def model_loss_train_freeze_attn(disp_ests, disp_gt, mask):
        weights = [0.5, 0.7, 1.0]
        all_losses = []
        for disp_est, weight in zip(disp_ests, weights):
            all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
        return sum(all_losses)

    @staticmethod
    def model_loss_train(disp_ests, disp_gt, mask):
        weights = [0.5, 0.5, 0.7, 1.0]
        all_losses = []
        for disp_est, weight in zip(disp_ests, weights):
            all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
        return sum(all_losses)

    @staticmethod
    def model_loss_test(disp_ests, disp_gt, mask):
        weights = [1.0]
        all_losses = []
        for disp_est, weight in zip(disp_ests, weights):
            all_losses.append(weight * F.l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
        return sum(all_losses)

    def __call__(self, training_output):
        disp_ests = training_output["disp"]["disp_ests"]
        disp_gt = training_output["disp"]["disp_gt"]
        mask = training_output["disp"]["mask"]
        assert self.loss_fn is not None, "loss_fn not initialized"
        total_loss = self.loss_fn(disp_ests, disp_gt, mask)
        loss_info = {}
        loss_info['scalar/train/loss_disp'] = total_loss
        loss_info['scalar/train/loss_sum'] = total_loss
        return total_loss, loss_info


class ACVNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_network(self):
        attn_weights_only = self.model_cfg['base_config']['attn_weights_only']
        freeze_attn_weights = self.model_cfg['base_config']['freeze_attn_weights']
        if self.model_cfg.get('small_model', False):
            self.net = NetSmall(self.max_disp, attn_weights_only, freeze_attn_weights)
        else:
            self.net = Net(self.max_disp, attn_weights_only, freeze_attn_weights)

    def build_loss_fn(self):
        if self.model_cfg['base_config']['attn_weights_only']:
            loss_type = 'attn_only'
        elif self.model_cfg['base_config']['freeze_attn_weights']:
            loss_type = 'freeze_attn'
        else:
            loss_type = 'full'
        self.loss_fn = ACVLoss(loss_type)

    def forward(self, inputs):
        """Forward the network."""
        ref_img = inputs["ref_img"]
        tgt_img = inputs["tgt_img"]
        if self.training:
            sequence_output = self.net(ref_img, tgt_img)
            res = sequence_output[-1]
            output = {
                "training_disp": {
                    "disp": {
                        "disp_ests": sequence_output,
                        "disp_gt": inputs['disp_gt'],
                        "mask": inputs['mask']
                    },
                },
                "visual_summary": {
                    "image/train/image_c": torch.cat([ref_img[0], tgt_img[0]], dim=1),
                    "image/train/disp_c": torch.cat(
                        [inputs["disp_gt"][0], res[0]],
                        dim=0
                    )
                }
            }
        else:
            pred2 = self.net(ref_img, tgt_img)
            res = pred2[-1]
            output = {
                "inference_disp": {
                    "disp_est": res
                },
                "visual_summary": {
                    "image/test/image_c": torch.cat([ref_img[0], tgt_img[0]], dim=1),
                    "image/test/disp_c": res[0]
                }
            }
            if 'disp_gt' in inputs:
                disp_gt = inputs['disp_gt']
                output['visual_summary'] = {
                    "image/val/image_c": torch.cat([ref_img[0], tgt_img[0]], dim=1),
                    "image/val/disp_c": torch.cat([disp_gt[0], res[0]], dim=0)
                }
        return output
