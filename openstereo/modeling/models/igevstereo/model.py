import torch
from torch.nn import functional as F

from modeling.base_model import BaseModel
from .igev_stereo import IGEVStereo
from .utils.augmentor import SparseFlowAugmentor, FlowAugmentor


class IGEVLoss:
    def __init__(self, loss_gamma=0.9, max_flow=192):
        self.loss_gamma=loss_gamma
        self.max_flow=max_flow
        self.info = {}

    """ Loss function defined over sequence of flow predictions """

    def sequenceloss(self, training_output):
        tarining_dsp = training_output['disp']
        disp_preds, flow_gt, valid = tarining_dsp['disp_ests'], tarining_dsp['disp_gt'], tarining_dsp['mask']
        loss_gamma = self.loss_gamma
        max_flow = self.max_flow
        
        flow_loss = 0.0

        # exlude invalid pixels and extremely large diplacements
        # mag = torch.sum(flow_gt**2, dim=1).sqrt()
        mag = flow_gt**2
        valid = ((valid >= 0.5) & (mag.sqrt() < max_flow)) # .unsqueeze(1)
        assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
        assert not torch.isinf(flow_gt[valid.bool()]).any()

        disp_init_pred = disp_preds[0]

        disp_gru_preds = disp_preds[1:]
        n_predictions = len(disp_gru_preds)
        assert n_predictions >= 1

        # flow_loss += 1.0 * F.smooth_l1_loss(disp_init_pred[valid.bool()], disp_gt[valid.bool()], size_average=True)
        #flow_loss += 1.0 * F.smooth_l1_loss(disp_init_pred[valid.bool()], flow_gt[valid.bool()], size_average=True)
        # print("disp_init_pred",disp_init_pred.shape)
        # print("flow_gt",flow_gt.shape)
        flow_loss += 1.0 * F.smooth_l1_loss(disp_init_pred.squeeze(1)[valid.bool()], flow_gt[valid.bool()], size_average=True)
        for i in range(n_predictions):
            assert not torch.isnan(disp_gru_preds[i]).any() and not torch.isinf(disp_gru_preds[i]).any()
            # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
            adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
            i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
            i_loss = (disp_gru_preds[i].squeeze(1) - flow_gt).abs()
            # assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
            flow_loss += i_weight * i_loss[valid.bool()].mean()

        # epe = (flow_preds[-1].squeeze(1) - flow_gt)**2

        epe = torch.abs(disp_gru_preds[-1].squeeze(1)[valid] - flow_gt[valid])
        # print(epe.shape)
        # epe = epe.view(-1)[valid.view(-1)].sqrt()
        # print(epe.shape)
        return flow_loss, epe

    def __call__(self, training_output):
        flow_loss, epe = self.sequenceloss(training_output)

        metrics = {
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
        }
        self.info['scalar/loss/sequenceloss'] = flow_loss.item()
        self.info['scalar/loss/epemean'] = epe.mean().item()
        self.info['scalar/loss/epe1px'] = (epe < 1).float().mean().item()
        self.info['scalar/loss/epe3px'] = (epe < 3).float().mean().item()
        self.info['scalar/loss/epe5px'] = (epe < 5).float().mean().item()

        return flow_loss, self.info


class IGEV_Stereo(BaseModel):
    def __init__(self, *args, **kwargs):
        self.sparse = False
        super().__init__(*args, **kwargs)

    def build_network(self):
        model_cfg = self.model_cfg
        if model_cfg['base_config']['aug_params'] is not None and "crop_size" in model_cfg['base_config']['aug_params']:
            aug_params = model_cfg['base_config']['aug_params']
            if self.sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)
        self.train_iters = model_cfg['base_config']['train_iters']
        # print(model_cfg['base_config'])
        self.net = IGEVStereo(model_cfg['base_config'])
        self.net.freeze_bn() # legacy, could be removed

    def build_loss_fn(self):
        self.loss_fn = IGEVLoss(loss_gamma=0.9, max_flow=192)
    
    def prepare_inputs(self, inputs, device=None):
        """Reorganize input data for different models

        Args:
            inputs: the input data.
        Returns:
            dict: training data including ref_img, tgt_img, disp image,
                  and other meta data.
        """
        # asure the disp_gt has the shape of [B, H, W]
        disp_gt = inputs['disp']
        if len(disp_gt.shape) == 4:
            disp_gt = disp_gt.squeeze(1)
        img1 = inputs['left'] 
        img2 = inputs['right']

        # compute the mask of valid disp_gt
        # valid = disp < 512
        # flow = torch.cat([-disp_gt, np.zeros_like(disp)], dim=-1)
        flow = disp_gt
        max_disp = self.model_cfg['base_config']['max_disp']
        mask = (disp_gt < max_disp) & (disp_gt > 0)

        """
        
        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 512) & (flow[1].abs() < 512)
        """

        return {
            'img1': img1,
            'img2': img2,
            'flow': disp_gt,
            'valid': mask,
            'disp_gt': disp_gt.unsqueeze(1).to(device),
            'mask': mask.unsqueeze(1).to(device),
        }

    def forward(self, inputs):
        """Forward the network."""
        img1 = inputs["img1"]
        img2 = inputs["img2"]
        if self.training:
            flow_predictions, flow = self.net(img1, img2, self.train_iters)
            output = {
                "training_disp": {
                    "disp": {
                        "disp_ests": flow_predictions,
                        "disp_gt": inputs['flow'],
                        "mask": inputs['valid']
                    },
                },
                "visual_summary": {},
            }
        else:
            flow = self.net(img1, img2, self.train_iters, test_mode = True)
            output = {
                "inference_disp": {
                    "disp_est": flow
                },
                "visual_summary": {}
            }
        return output
