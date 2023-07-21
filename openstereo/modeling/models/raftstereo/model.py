import torch

from modeling.base_model import BaseModel
from .raft_stereo import RAFTStereo
from .utils.augmentor import SparseFlowAugmentor, FlowAugmentor


class RAFTLoss:
    def __init__(self, loss_gamma=0.9, max_flow=700):
        self.loss_gamma=loss_gamma
        self.max_flow=max_flow
        self.info = {}

    """ Loss function defined over sequence of flow predictions """

    def sequenceloss(self, training_output):
        tarining_dsp = training_output['disp']
        flow_preds, flow_gt, valid = tarining_dsp['disp_ests'], tarining_dsp['disp_gt'], tarining_dsp['mask']
        loss_gamma = self.loss_gamma
        max_flow = self.max_flow
        n_predictions = len(flow_preds)
        assert n_predictions >= 1
        flow_loss = 0.0

        # exlude invalid pixels and extremely large diplacements
        # mag = torch.sum(flow_gt**2, dim=1).sqrt()
        mag = flow_gt**2

        # exclude extremly large displacements
        # print('flow_gt {}'.format(flow_gt.shape))
        # print('mag {}'.format(mag.shape))
        # print('valid {}'.format(valid.shape))
        valid = ((valid >= 0.5) & (mag.sqrt() < max_flow)) # .unsqueeze(1)
        assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
        assert not torch.isinf(flow_gt[valid.bool()]).any()

        for i in range(n_predictions):
            assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
            # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
            adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
            i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
            i_loss = (flow_preds[i].squeeze(1) - flow_gt).abs()
            # assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
            flow_loss += i_weight * i_loss[valid.bool()].mean()

        # epe = (flow_preds[-1].squeeze(1) - flow_gt)**2

        epe = torch.abs(flow_preds[-1].squeeze(1)[valid] - flow_gt[valid])
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


class RAFT_Stereo(BaseModel):
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
        self.net = RAFTStereo(model_cfg['base_config'])
        self.net.freeze_bn() # legacy, could be removed

    def build_loss_fn(self):
        self.loss_fn = RAFTLoss(loss_gamma=0.9, max_flow=700)

    def prepare_inputs(self, inputs, device=None, **kwargs):
        """Reorganize input data for different models

        Args:
            inputs: the input data.
            device: the device to put the data.
        Returns:
            dict: training data including ref_img, tgt_img, disp image,
                  and other meta data.
        """
        processed_inputs = {
            'img1': inputs['left'],
            'img2': inputs['right'],
        }
        if 'disp' in inputs.keys():
            disp_gt = inputs['disp']
            # compute the mask of valid disp_gt
            mask = (disp_gt < self.max_disp) & (disp_gt > 0)
            processed_inputs.update({
                'flow': disp_gt,
                'valid': mask,
                'disp_gt': disp_gt,
                'mask': mask            
            })
        if not self.training:
            for k in ['pad', 'name']:
                if k in inputs.keys():
                    processed_inputs[k] = inputs[k]
        if device is not None:
            # move data to device
            for k, v in processed_inputs.items():
                processed_inputs[k] = v.to(device) if torch.is_tensor(v) else v
        return processed_inputs

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
                        "disp_gt": inputs['flow'],
                        "mask": inputs['valid']
                    },
                },
                "visual_summary": {},
            }
        else:
            cor_dis, flow = self.net(img1, img2, self.train_iters, test_mode = True)
            output = {
                "inference_disp": {
                    "disp_est": flow[-1].squeeze(1)
                },
                "visual_summary": {}
            }
        return output
