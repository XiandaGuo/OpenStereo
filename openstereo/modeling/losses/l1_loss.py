import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper


class Smooth_l1_Loss(BaseLoss):
    def __init__(self, size_average=True):
        super(Smooth_l1_Loss, self).__init__(size_average)
        self.size_average = size_average

    def forward(self, logits, labels):
        return F.smooth_l1_loss(logits, labels, size_average=self.size_average)


class Weighted_Smooth_l1_Loss(BaseLoss):
    def __init__(self, weights=None, reduction='mean'):
        super().__init__(weights)
        self.weights = [0.5, 0.5, 0.7, 1.0] if weights is None else weights
        self.reduction = reduction

    @gather_and_scale_wrapper
    def forward(self, disp_ests, disp_gt, mask=None):
        weights = self.weights
        all_losses = []
        for disp_est, weight in zip(disp_ests, weights):
            all_losses.append(
                weight * F.smooth_l1_loss(
                    disp_est[mask] if mask is not None else disp_est,
                    disp_gt[mask] if mask is not None else disp_gt,
                    reduction=self.reduction
                )
            )
        return sum(all_losses)

        # all_losses = []
        # if mask is None:
        #     for disp_est, weight in zip(logits, self.weights):
        #         all_losses.append(weight * F.smooth_l1_loss(disp_est, labels, reduction=self.reduction))
        # else:
        #     for disp_est, weight in zip(logits, self.weights):
        #         all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], labels[mask], reduction=self.reduction))
        # return sum(all_losses)
