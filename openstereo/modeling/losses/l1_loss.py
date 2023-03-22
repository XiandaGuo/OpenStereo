import torch
import torch.nn as nn
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
        self.info.update({'loss': loss})
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
        self.info.update({'loss': loss})
        return loss, self.info
    

class MultiScaleLoss(BaseLoss):
    def __init__(self, loss_term_weight=1.0, scales=7, downscale=1, weights=None, loss='L1', maxdisp=192, mask=False):
        super().__init__(loss_term_weight)
        self.downscale = downscale
        self.mask = mask
        self.maxdisp = maxdisp
        if weights is None:
            print("MultiScaleLoss's weights none")
        self.weights=torch.Tensor(weights).cuda()
        #self.weights = torch.Tensor(scales).fill_(1).cuda() if weights is None else torch.Tensor(weights).cuda()
        assert(len(self.weights[0]) == scales)

        if type(loss) is str:

            if loss == 'L1':
                self.loss = nn.L1Loss()
            elif loss == 'MSE':
                self.loss = nn.MSELoss()
            elif loss == 'SmoothL1':
                self.loss = nn.SmoothL1Loss()
            elif loss == 'MAPE':
                self.loss = nn.MAPELoss()
        else:
            self.loss = loss
        self.multiScales = [nn.AvgPool2d(self.downscale*(2**i), self.downscale*(2**i)) for i in range(scales)]
       
    def forward(self, disp_ests, disp_gt, mask=None,round=0):
        weights=self.weights[round]
       
        if (type(disp_ests) is tuple) or (type(disp_ests) is list):
            out = 0
            
            for i, input_ in enumerate(disp_ests):
                target_ = self.multiScales[i](disp_gt)
               
                if input_.dim()==4:
                    input_=input_.squeeze(1)
                EPE_ = self.SL_EPE(input_, target_, self.maxdisp)
                out += weights[i] * EPE_
        else:
            out = self.loss(disp_ests, self.multiScales[0](disp_gt))
        self.info.update({'loss': out})
        return out, self.info
    
    def SL_EPE(self,input_flow, target_flow, maxdisp=192):
        target_valid = (target_flow < maxdisp) & (target_flow > 0)
        return F.smooth_l1_loss(input_flow[target_valid], target_flow[target_valid], reduction='mean')


class MultiAANetScaleLoss(BaseLoss):
    def __init__(self, loss_term_weight=1.0, scales=7, downscale=1, weights=None, loss='L1', maxdisp=192, mask=False):
        super().__init__(loss_term_weight)
        self.downscale = downscale
        self.mask = mask
        self.maxdisp = maxdisp
        self.weights = torch.Tensor(scales).fill_(1).cuda() if weights is None else torch.Tensor(weights).cuda()
        assert(len(self.weights[0]) == scales)

        if type(loss) is str:

            if loss == 'L1':
                self.loss = nn.L1Loss()
            elif loss == 'MSE':
                self.loss = nn.MSELoss()
            elif loss == 'SmoothL1':
                self.loss = nn.SmoothL1Loss()
            elif loss == 'MAPE':
                self.loss = nn.MAPELoss()
        else:
            self.loss = loss
        self.multiScales = [nn.AvgPool2d(self.downscale*(2**i), self.downscale*(2**i)) for i in range(scales)]
       
    def forward(self, disp_ests, disp_gt, mask=None,round=0):
       # weights=self.weights[round]
        weights=[1/3,2/3,1.0,1.0,1.0]
        if (type(disp_ests) is tuple) or (type(disp_ests) is list):
            out = 0
        
            for i, input_ in enumerate(disp_ests):
                #print("input_",input_.shape)
             
                input_=torch.unsqueeze(input_, 1)
                #print("input_",input_.shape)
                input_ = F.interpolate(input_, size=(disp_gt.size(-2), disp_gt.size(-1)), mode='bilinear', align_corners=False) * (disp_gt.size(-1) / input_.size(-1))
                #if input_.dim()==4:
                input_=input_.squeeze(1)
                
                EPE_ = self.SL_EPE(input_, disp_gt, self.maxdisp)
                out += weights[i] * EPE_
        else:
            return 
        self.info.update({'loss': out})
        return out, self.info
    
    def SL_EPE(self,input_flow, target_flow, maxdisp=192):
        target_valid = (target_flow < maxdisp) & (target_flow > 0)
        target_valid.detach_()
        return F.smooth_l1_loss(input_flow[target_valid], target_flow[target_valid], size_average=True)