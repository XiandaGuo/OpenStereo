import torch.nn.functional as F

from .base import BaseLoss


class Smooth_l1_Loss(BaseLoss):
    def __init__(self, reduction='mean', loss_term_weight=1.0):
        super().__init__(loss_term_weight)
        self.reduction = reduction

    def forward(self, disp_ests, disp_gt, mask=None):
        loss = F.smooth_l1_loss(
            disp_ests[mask] if mask is not None else disp_ests,
            disp_gt[mask] if mask is not None else disp_gt,
            reduction=self.reduction
        )
        # self.info.update({'loss': loss})
        return loss, self.info


class Weighted_Smooth_l1_Loss(BaseLoss):
    def __init__(self, weights, reduction='mean', loss_term_weight=1.0):
        super().__init__(loss_term_weight)
        self.weights = weights
        self.reduction = reduction

    def forward(self, disp_ests, disp_gt, mask=None):
        weights = self.weights
        loss = 0.
        for disp_est, weight in zip(disp_ests, weights):
            loss += weight * F.smooth_l1_loss(
                disp_est[mask] if mask is not None else disp_est,
                disp_gt[mask] if mask is not None else disp_gt,
                reduction=self.reduction
            )
        # self.info.update({'loss': loss})
        return loss, self.info
