import torch
import torch.nn as nn
import torch.nn.functional as F


class FasterSoftArgmin(nn.Module):
    """
    A faster implementation of soft argmin.
    details can refer to dmb.modeling.stereo.disp_predictors.soft_argmin
    Args:
        max_disp, (int): under the scale of feature used,
            often equals to (end disp - start disp + 1), the maximum searching range of disparity
        start_disp (int): the start searching disparity index, usually be 0
        dilation (optional, int): the step between near disparity index
        alpha (float or int): a factor will times with cost_volume
            details can refer to: https://bouthilx.wordpress.com/2013/04/21/a-soft-argmax/
        normalize (bool): whether apply softmax on cost_volume, default True

    Inputs:
        cost_volume (Tensor): the matching cost after regularization,
            in [BatchSize, disp_sample_number, Height, Width] layout
        disp_sample (optional, Tensor): the estimated disparity samples,
            in [BatchSize, disp_sample_number, Height, Width] layout. NOT USED!
    Returns:
        disp_map (Tensor): a disparity map regressed from cost volume,
            in [BatchSize, 1, Height, Width] layout
    """

    def __init__(self, max_disp, start_disp=0, dilation=1, alpha=1.0, normalize=True):
        super(FasterSoftArgmin, self).__init__()
        self.max_disp = max_disp
        self.start_disp = start_disp
        self.dilation = dilation
        self.end_disp = start_disp + max_disp - 1
        self.disp_sample_number = (max_disp + dilation - 1) // dilation

        self.alpha = alpha
        self.normalize = normalize

        # compute disparity index: (1 ,1, disp_sample_number, 1, 1)
        disp_sample = torch.linspace(
            self.start_disp, self.end_disp, self.disp_sample_number
        )
        disp_sample = disp_sample.repeat(1, 1, 1, 1, 1).permute(0, 1, 4, 2, 3).contiguous()

        self.disp_regression = nn.Conv3d(1, 1, (self.disp_sample_number, 1, 1), 1, 0, bias=False)

        self.disp_regression.weight.data = disp_sample
        self.disp_regression.weight.requires_grad = False

    def forward(self, cost_volume):

        # note, cost volume direct represent similarity
        # 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.

        if cost_volume.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(cost_volume.dim()))

        # scale cost volume with alpha
        cost_volume = cost_volume * self.alpha

        if self.normalize:
            prob_volume = F.softmax(cost_volume, dim=1)
        else:
            prob_volume = cost_volume

        # [B, disp_sample_number, W, H] -> [B, 1, disp_sample_number, W, H]
        prob_volume = prob_volume.unsqueeze(1)

        disp_map = self.disp_regression(prob_volume)
        # [B, 1, 1, W, H] -> [B, W, H]
        disp_map = disp_map.squeeze(1).squeeze(1)
        return disp_map

    def __repr__(self):
        repr_str = '{}\n'.format(self.__class__.__name__)
        repr_str += ' ' * 4 + 'Max Disparity: {}\n'.format(self.max_disp)
        repr_str += ' ' * 4 + 'Start disparity: {}\n'.format(self.start_disp)
        repr_str += ' ' * 4 + 'Dilation rate: {}\n'.format(self.dilation)
        repr_str += ' ' * 4 + 'Alpha: {}\n'.format(self.alpha)
        repr_str += ' ' * 4 + 'Normalize: {}\n'.format(self.normalize)

        return repr_str

    @property
    def name(self):
        return 'FasterSoftArgmin'


class PSMDispProcessor(nn.Module):
    def __init__(self, max_disp=192):
        super().__init__()
        self.disp_processor = FasterSoftArgmin(
            # the maximum disparity of disparity search range
            max_disp=max_disp,
            # the start disparity of disparity search range
            start_disp=0,
            # the step between near disparity sample
            dilation=1,
            # the temperature coefficient of soft argmin
            alpha=1.0,
            # whether normalize the estimated cost volume
            normalize=True
        )

    def forward(self, inputs):
        cost1 = inputs['cost1']
        cost2 = inputs['cost2']
        cost3 = inputs['cost3']
        ref_img = inputs['ref_img']
        tgt_img = inputs['tgt_img']

        disp1 = self.disp_processor(cost1)
        disp2 = self.disp_processor(cost2)
        disp3 = self.disp_processor(cost3)
        if self.training:
            disp_gt = inputs['disp_gt']
            output = {
                "training_disp": {
                    "disp": {
                        "disp_ests": [disp1, disp2, disp3],
                        "disp_gt": inputs['disp_gt'],
                        "mask": inputs['mask']
                    },
                },
                "visual_summary": {
                    'image/train/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                    'image/train/disp_c': torch.cat([disp_gt[0], disp3[0]], dim=0),
                },
            }
        else:
            output = {
                "inference_disp": {
                    "disp_est": disp3,
                },
                "visual_summary": {
                    'image/test/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                    'image/test/disp_c': disp3[0]
                }
            }
            if 'disp_gt' in inputs:
                disp_gt = inputs['disp_gt']
                output['visual_summary'] = {
                    'image/val/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                    'image/val/disp_c': torch.cat([disp_gt[0], disp3[0]], dim=0),
                }
        return output
