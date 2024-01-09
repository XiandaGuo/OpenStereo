import torch
import torch.optim as optim

from modeling.base_model import BaseModel
from utils import get_valid_args, get_attr_from
from .utilities.loss import build_criterion
from .sttr import STTR
from .utilities import Map
from .utilities.misc import NestedTensor


class STTRLoss:
    def __init__(self, loss_cfg):
        self.info = {}
        loss_cfg = Map(loss_cfg)
        self.criterion = build_criterion(loss_cfg)

    """ Loss function defined over sequence of flow predictions """

    def __call__(self, training_output):
        outputs = training_output["disp"]["outputs"]
        inputs = training_output["disp"]["inputs"]
        # print(outputs)
        losses = self.criterion(inputs, outputs)
        if losses is None:
            return None
        # print(losses.keys())
        # for key in losses.keys():
        #     print("{} value is {}".format(key, losses[key]))
        total_loss = losses['aggregated']

        self.info['scalar/loss/aggregated'] = losses['aggregated'].item()
        self.info['scalar/loss/rr'] = losses['rr'].item()
        self.info['scalar/loss/l1_raw'] = losses['l1_raw'].item()
        self.info['scalar/loss/l1'] = losses['l1'].item()
        self.info['scalar/loss/occ_be'] = losses['occ_be'].item()
        self.info['scalar/loss/error_px'] = losses['error_px']
        self.info['scalar/loss/total_px'] = losses['total_px']
        self.info['scalar/loss/epe'] = losses['epe']
        self.info['scalar/loss/iou'] = losses['iou']

        return total_loss, self.info


class STTRNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_network(self):
        model_cfg = self.model_cfg
        maxdisp = model_cfg['base_config']['max_disp']
        args = Map(model_cfg['base_config'])
        self.downsample = model_cfg['base_config']['downsample']
        self.net = STTR(args)
        # self.net.freeze_bn() # legacy, could be removed

    def build_loss_fn(self):
        loss_cfg = self.cfg['loss_cfg']
        self.loss_fn = STTRLoss(loss_cfg)

    def get_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        optimizer = get_attr_from([optim], optimizer_cfg['solver'])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])
        args = Map(optimizer_cfg)
        param_dicts = [
            {"params": [p for n, p in self.net.named_parameters() if
                        "backbone" not in n and "regression" not in n and p.requires_grad],
             "lr": args.lr, },
            {
                "params": [p for n, p in self.net.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
            {
                "params": [p for n, p in self.net.named_parameters() if "regression" in n and p.requires_grad],
                "lr": args.lr_regression,
            },
        ]
        optimizer = optimizer(params=param_dicts, **valid_arg)
        return optimizer

    # def prepare_inputs(self, inputs, device=None):
    #     """Reorganize input data for different models
    #
    #     Args:
    #         inputs: the input data.
    #     Returns:
    #         dict: training data including ref_img, tgt_img, disp image,
    #               and other meta data.
    #     """
    #     # asure the disp_gt has the shape of [B, H, W]
    #     disp_gt = inputs['disp']
    #     if len(disp_gt.shape) == 4:
    #         disp_gt = disp_gt.squeeze(1)
    #
    #     # compute the mask of valid disp_gt
    #     max_disp = self.cfgs['model_cfg']['base_config']['max_disp']
    #     mask = (disp_gt < max_disp) & (disp_gt > 0)
    #     # fake occ
    #
    #     return {
    #         'left': inputs['left'],
    #         'right': inputs['right'],
    #         'disp_gt': disp_gt,
    #         'mask': mask,
    #     }

    def forward(self, inputs):
        """Forward the network."""
        # fake occ
        # print(inputs.keys())
        if "occ_mask" in inputs.keys():
            occ_mask = inputs['occ_mask']
            occ_mask_right = inputs['occ_mask_right']
            # change occ_mask tensor type to long or bool type
            occ_mask = occ_mask.bool()
            occ_mask_right = occ_mask_right.bool()
        else:
            occ_mask = inputs['mask']
            occ_mask_right = inputs['mask']
        # read data
        left, right = inputs['ref_img'], inputs['tgt_img']
        disp, occ_mask, occ_mask_right = inputs['disp_gt'], occ_mask, occ_mask_right

        # if need to downsample, sample with a provided stride
        device = left.get_device()
        downsample = self.downsample
        bs, _, h, w = left.size()
        if downsample <= 0:
            sampled_cols = None
            sampled_rows = None
        else:
            col_offset = int(downsample / 2)
            row_offset = int(downsample / 2)
            sampled_cols = torch.arange(col_offset, w, downsample)[None,].expand(bs, -1).to(device)
            sampled_rows = torch.arange(row_offset, h, downsample)[None,].expand(bs, -1).to(device)

        # build the input
        inputs_tensor = NestedTensor(left, right, sampled_cols=sampled_cols, sampled_rows=sampled_rows, disp=disp,
                                     occ_mask=occ_mask, occ_mask_right=occ_mask_right)
        # input_data['occ_mask'] = np.zeros_like(disp).astype(np.bool)
        # currently fake occ for debug warting for juntao.lu add data streaming

        if self.training:
            outputs = self.net(inputs_tensor)
            output = {
                "training_disp": {
                    "disp": {
                        "disp_ests": outputs['disp_pred'],
                        "disp_gt": inputs['disp_gt'],
                        "mask": inputs['mask'],
                        "inputs": inputs_tensor,
                        "outputs": outputs
                    },
                },
                "visual_summary": {},
            }
        else:
            inputs_tensor = NestedTensor(left, right, sampled_cols=sampled_cols, sampled_rows=sampled_rows, disp=disp,
                                     occ_mask=occ_mask, occ_mask_right=occ_mask_right)
            outputs = self.net(inputs_tensor)
            output = {
                "inference_disp": {
                    "disp_est": outputs['disp_pred']
                },
                "visual_summary": {}
            }
        return output
