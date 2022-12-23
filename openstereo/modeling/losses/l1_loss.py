import torch
import torch.nn.functional as F

from .base import BaseLoss

class Smooth_l1_Loss(BaseLoss):
    def __init__(self,size_average=True):
        super(Smooth_l1_Loss, self).__init__(size_average)
        self.size_average=size_average

    def forward(self, logits, labels):
            return F.smooth_l1_loss(logits,labels,size_average=self.size_average)