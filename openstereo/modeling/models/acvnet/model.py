import torch.nn.functional as F
import torch 

from modeling.base_model import BaseModel
from .raft_stereo import RAFTStereo


class RAFTLoss:
    def __init__(self, loss_gamma=0.9, max_flow=700):
        self.loss_gamma=loss_gamma
        self.max_flow=max_flow
        self.info = {}

    """ Loss function defined over sequence of flow predictions """

    def sequenceloss(self, training_output):
        print(training_output.keys())
        flow_preds, flow_gt, valid = training_output['disp_ests'], training_output['disp_gt'], training_output['mask']
        loss_gamma = self.loss_gamma
        max_flow = self.max_flow
        n_predictions = len(flow_preds)
        assert n_predictions >= 1
        flow_loss = 0.0

        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt**2, dim=1).sqrt()

        # exclude extremly large displacements
        valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
        assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
        assert not torch.isinf(flow_gt[valid.bool()]).any()

        for i in range(n_predictions):
            assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
            # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
            adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
            i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
            i_loss = (flow_preds[i] - flow_gt).abs()
            assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
            flow_loss += i_weight * i_loss[valid.bool()].mean()

        epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]
        
        return flow_loss, epe

    def __call__(self, training_output):
        flow_loss, epe = self.sequenceloss(training_output)

        metrics = {
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
        }
        self.info['scalar/epemean'] = epe.mean().item()
        self.info['scalar/epe1px'] = (epe < 1).float().mean().item()
        self.info['scalar/epe3px'] = (epe < 3).float().mean().item()
        self.info['scalar/epe5px'] = (epe < 5).float().mean().item()

        return flow_loss, self.info


class ACVNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_network(self, model_cfg):
        if model_cfg['base_config']['aug_params'] is not None and "crop_size" in model_cfg['base_config']['aug_params']:
            aug_params = model_cfg['base_config']['aug_params']
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)
        self.train_iters = model_cfg['base_config']['train_iters']
        self.net = RAFTStereo(model_cfg['base_config'])
        self.net.freeze_bn() # legacy, could be removed

    def get_loss_func(self, loss_cfg):
        return RAFTLoss(loss_gamma=0.9, max_flow=700)
    
    def inputs_pretreament(self, inputs):
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
        max_disp = self.cfgs['model_cfg']['base_config']['max_disp']
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
        }

    def forward(self, inputs):
        """Forward the network."""
        img1 = inputs["img1"]
        img2 = inputs["img2"]
        if self.training:
            flow_predictions, cor_dis, flow = self.net(img1, img2, self.train_iters)
            output = {
                "training_disp": {
                    "disp": {
                        "disp_ests": flow_predictions,
                        "disp_gt": inputs['disp_gt'],
                        "mask": inputs['mask']
                    },
                },
                "visual_summary": {},
            }
        else:
            cor_dis, flow = self.net(rimg1, img2, self.train_iters, test_mode = True)
            output = {
                "inference_disp": {
                    "disp_est": flow
                },
                "visual_summary": {}
            }
        return output
