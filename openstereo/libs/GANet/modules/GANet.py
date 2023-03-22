from torch.nn.modules.module import Module
import torch
import numpy as np
from torch.autograd import Variable
from ..functions import *
import torch.nn.functional as F

from ..functions.GANet import MyLossFunction
from ..functions.GANet import SgaFunction
from ..functions.GANet import NlfFunction
from ..functions.GANet import LgaFunction
from ..functions.GANet import Lga2Function
from ..functions.GANet import Lga3Function
from ..functions.GANet import Lga3dFunction
from ..functions.GANet import Lga3d2Function
from ..functions.GANet import Lga3d3Function
from ..functions.GANet import MyLoss2Function
from ..functions.GANet import NlfDownFunction
from ..functions.GANet import NlfUpFunction
from ..functions.GANet import NlfRightFunction
from ..functions.GANet import NlfLeftFunction
class ChannelNorm(Module):
    def __init__(self, eps=1e-5):
        super(ChannelNorm, self).__init__()
#        self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
#        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
#        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
#        N,C,H,W = x.size()
#        G = self.num_groups

#        x = x.view(N,G,-1)
        mean = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        return x
#        x = x.view(N,C,H,W)
#        return x * self.weight + self.bias

class GetWeights(Module):
    def __init__(self, wsize=5):
        super(GetWeights, self).__init__()
        self.wsize = wsize

#        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(self.maxdisp)),[1,self.maxdisp,1,1])).cuda(), requires_grad=False)
    def forward(self, x):
        assert(x.is_contiguous() == True)
        with torch.cuda.device_of(x):
#            x = F.normalize(x, p=2, dim=1)
            num, channels, height, width = x.size()

            weight_down = x.new().resize_(num, 5, height, width).zero_()
            weight_down[:, 0, :, :] = torch.sum(x * x, 1)
            weight_down[:, 1, 1:, :] = torch.sum(x[:, :, 1:, :] * x[:, :, :-1, :], 1)
            weight_down[:, 2, 1:, 1:] = torch.sum(x[:, :, 1:, 1:] * x[:, :, :-1, :-1], 1)
            weight_down[:, 3, 1:, :-1] = torch.sum(x[:, :, 1:, :-1] * x[:, :, :-1, 1:], 1)
            weight_down[:, 4, :, 1:] = torch.sum(x[:, :, :, 1:] * x[:, :, :, :-1], 1)

            weight_up = x.new().resize_(num, 5, height, width).zero_()
            weight_up[:, 0, :, :] = torch.sum(x * x, 1)
            weight_up[:, 1, :-1, :] = torch.sum(x[:, :, :-1, :] * x[:, :, 1:, :], 1)
            weight_up[:, 2, :-1, 1:] = torch.sum(x[:, :, :-1, 1:] * x[:, :, 1:, :-1], 1)
            weight_up[:, 3, :-1, :-1] = torch.sum(x[:, :, :-1, :-1] * x[:, :, 1:, 1:], 1)
            weight_up[:, 4, :, :-1] = torch.sum(x[:, :, :, :-1] * x[:, :, :, 1:], 1)

            weight_right = x.new().resize_(num, 5, height, width).zero_()
            weight_right[:, 0, :, :] = torch.sum(x * x, 1)
            weight_right[:, 1, :, 1:] = torch.sum(x[:, :, :, 1:] * x[:, :, :, :-1], 1)
            weight_right[:, 2, 1:, 1:] = torch.sum(x[:, :, 1:, 1:] * x[:, :, :-1, :-1], 1)
            weight_right[:, 3, :-1, 1:] = torch.sum(x[:, :, :-1, 1:] * x[:, :, 1:, :-1], 1)
            weight_right[:, 4, 1:, :] = torch.sum(x[:, :, 1:, :] * x[:, :, :-1, :], 1)

            weight_left = x.new().resize_(num, 5, height, width).zero_()
            weight_left[:, 0, :, :] = torch.sum(x * x, 1)
            weight_left[:, 1, :, :-1] = torch.sum(x[:, :, :, :-1] * x[:, :, :, 1:], 1)
            weight_left[:, 2, 1:, :-1] = torch.sum(x[:, :, 1:, :-1] * x[:, :, :-1, 1:], 1)
            weight_left[:, 3, :-1, :-1] = torch.sum(x[:, :, :-1, :-1] * x[:, :, 1:, 1:], 1)
            weight_left[:, 4, :-1, :] = torch.sum(x[:, :, :-1, :] * x[:, :, 1:, :], 1)
            
#            weight_down = F.normalize(weight_down, p=1, dim=1)
#            weight_up = F.normalize(weight_up, p=1, dim=1)
#            weight_right = F.normalize(weight_right, p=1, dim=1)
#            weight_left = F.normalize(weight_left, p=1, dim=1)

            weight_down = F.softmax(weight_down, dim=1)
            weight_up = F.softmax(weight_up, dim=1)
            weight_right = F.softmax(weight_right, dim=1)
            weight_left = F.softmax(weight_left, dim=1)

            weight_down = weight_down.contiguous()
            weight_up = weight_up.contiguous()
            weight_right = weight_right.contiguous()
            weight_left = weight_left.contiguous()
         
        return weight_down, weight_up, weight_right, weight_left

class GetFilters(Module):
    def __init__(self, radius=2):
        super(GetFilters, self).__init__()
        self.radius = radius
        self.wsize = (radius*2 + 1) * (radius*2 + 1)
#        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(self.maxdisp)),[1,self.maxdisp,1,1])).cuda(), requires_grad=False)
    def forward(self, x):
        assert(x.is_contiguous() == True)
   
        with torch.cuda.device_of(x):
            x = F.normalize(x, p=2, dim=1)
#            num, channels, height, width = x.size()
#            rem = torch.unsqueeze(x, 2).repeat(1, 1, self.wsize, 1, 1)
#
#            temp = x.new().resize_(num, channels, self.wsize, height, width).zero_()
#            idx = 0
 #           for r in range(-self.radius, self.radius+1):
 #               for c in range(-self.radius, self.radius+1):
 #                   temp[:, :, idx, max(-r, 0):min(height - r, height), max(-c,0):min(width-c, width)] = x[:, :, max(r, 0):min(height + r, height), max(c, 0):min(width + c, width)]
#                    idx += 1    
#            filters = torch.squeeze(torch.sum(rem*temp, 1), 1)
#            filters = F.normalize(filters, p=1, dim=1)
#            filters = filters.contiguous()
#        return filters

  
            num, channels, height, width = x.size()
            filters = x.new().resize_(num, self.wsize, height, width).zero_()
            idx = 0
            for r in range(-self.radius, self.radius+1):
                for c in range(-self.radius, self.radius+1):
                    filters[:, idx, max(-r, 0):min(height - r, height), max(-c,0):min(width-c, width)] = torch.squeeze(torch.sum(x[:, :, max(r, 0):min(height + r, height), max(c, 0):min(width + c, width)] * x[:,:,max(-r, 0):min(height-r, height), max(-c,0):min(width-c,width)], 1),1)
                    idx += 1      
       
            filters = F.normalize(filters, p=1, dim=1)
            filters = filters.contiguous()
        return filters

         

 

class MyNormalize(Module):
    def __init__(self, dim):
        self.dim = dim
        super(MyNormalize, self).__init__()
    def forward(self, x):
#        assert(x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            norm = torch.sum(torch.abs(x),self.dim)
            norm[norm <= 0] = norm[norm <= 0] - 1e-6
            norm[norm >= 0] = norm[norm >= 0] + 1e-6
            norm = torch.unsqueeze(norm, self.dim)
            size = np.ones(x.dim(), dtype='int')
            size[self.dim] = x.size()[self.dim]
            norm = norm.repeat(*size)
            x = torch.div(x, norm)
        return x
class MyLoss2(Module):
    def __init__(self, thresh=1, alpha=2):
        super(MyLoss2, self).__init__()
        self.thresh = thresh
        self.alpha = alpha
    def forward(self, input1, input2):
        result = MyLoss2Function.apply(input1, input2, self.thresh, self.alpha)
        return result
class MyLoss(Module):
    def __init__(self, upper_thresh=5, lower_thresh=1):
        super(MyLoss, self).__init__()
        self.upper_thresh = 5
        self.lower_thresh = 1
    def forward(self, input1, input2):
        result = MyLossFunction.apply(input1, input2, self.upper_thresh, self.lower_thresh)
        return result
class NLFMax(Module):
    def __init__(self):
        super(NLFMax, self).__init__()
    def forward(self, input, g0, g1, g2, g3):
        result0 = NlfDownFunction.apply(input, g0)
        result1 = NlfUpFunction.apply(input, g1)
        result2 = NlfRightFunction.apply(input, g2)
        result3 = NlfLeftFunction.apply(input, g3)
        return torch.max(torch.max(torch.max(result0, result1), result2), result3)

class NLFMean(Module):
    def __init__(self):
        super(NLFMean, self).__init__()
    def forward(self, input, g0, g1, g2, g3):
        result0 = NlfDownFunction.apply(input, g0)
        result1 = NlfUpFunction.apply(input, g1)
        result2 = NlfRightFunction.apply(input, g2)
        result3 = NlfLeftFunction.apply(input, g3)
#        result1 = NlfUpFunction()(input, g1)
#        result2 = NlfRightFunction()(input, g2)
#        result3 = NlfLeftFunction()(input, g3)
#        return torch.add(torch.add(torch.add(result0, result1), result2), result3)
        return (result0 + result1 + result2 + result3) * 0.25
class NLFIter(Module):
    def __init__(self):
        super(NLFIter, self).__init__()
    def forward(self, input, g0, g1, g2, g3):
        result = NlfDownFunction.apply(input, g0)
        result = NlfUpFunction.apply(result, g1)
        result = NlfRightFunction.apply(result, g2)
        result = NlfLeftFunction.apply(result, g3)
        return result
class NLF(Module):
    def __init__(self):
        super(NLF, self).__init__()

    def forward(self, input, g0, g1, g2, g3):
        result = NlfFunction.apply(input, g0, g1, g2, g3)
        return result
class SGA(Module):
    def __init__(self):
        super(SGA, self).__init__()

    def forward(self, input, g0, g1, g2, g3):
        result = SgaFunction.apply(input, g0, g1, g2, g3)
        return result
		

		
class LGA3D3(Module):
    def __init__(self, radius=2):
        super(LGA3D3, self).__init__()
        self.radius = radius

    def forward(self, input1, input2):
        result = Lga3d3Function.apply(input1, input2, self.radius)
        return result
class LGA3D2(Module):
    def __init__(self, radius=2):
        super(LGA3D2, self).__init__()
        self.radius = radius

    def forward(self, input1, input2):
        result = Lga3d2Function.apply(input1, input2, self.radius)
        return result
class LGA3D(Module):
    def __init__(self, radius=2):
        super(LGA3D, self).__init__()
        self.radius = radius

    def forward(self, input1, input2):
        result = Lga3dFunction.apply(input1, input2, self.radius)
        return result		
		
class LGA3(Module):
    def __init__(self, radius=2):
        super(LGA3, self).__init__()
        self.radius = radius

    def forward(self, input1, input2):
        result = Lga3Function.apply(input1, input2, self.radius)
        return result
class LGA2(Module):
    def __init__(self, radius=2):
        super(LGA2, self).__init__()
        self.radius = radius

    def forward(self, input1, input2):
        result = Lga2Function.apply(input1, input2, self.radius)
        return result
class LGA(Module):
    def __init__(self, radius=2):
        super(LGA, self).__init__()
        self.radius = radius

    def forward(self, input1, input2):
        result = LgaFunction.apply(input1, input2, self.radius)
        return result
		


class GetCostVolume(Module):
    def __init__(self, maxdisp):
        super(GetCostVolume, self).__init__()
        self.maxdisp=maxdisp+1

    def forward(self, x,y):
        assert(x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            num, channels, height, width = x.size()
            cost = x.new().resize_(num, channels*2, self.maxdisp, height, width).zero_()
#            cost = Variable(torch.FloatTensor(x.size()[0], x.size()[1]*2, self.maxdisp,  x.size()[2],  x.size()[3]).zero_(), volatile= not self.training).cuda()
            for i in range(self.maxdisp):
                if i > 0 :
                    cost[:, :x.size()[1], i, :,i:]   = x[:,:,:,i:]
                    cost[:, x.size()[1]:, i, :,i:]   = y[:,:,:,:-i]
                else:
                    cost[:, :x.size()[1], i, :,:]   = x
                    cost[:, x.size()[1]:, i, :,:]   = y

            cost = cost.contiguous()
        return cost
 
class DisparityRegression(Module):
    def __init__(self, maxdisp):
        super(DisparityRegression, self).__init__()
        self.maxdisp = maxdisp+1
#        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(self.maxdisp)),[1,self.maxdisp,1,1])).cuda(), requires_grad=False)

    def forward(self, x):
        assert(x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            disp = Variable(torch.Tensor(np.reshape(np.array(range(self.maxdisp)),[1,self.maxdisp,1,1])).cuda(), requires_grad=False)
            disp = disp.repeat(x.size()[0],1,x.size()[2],x.size()[3])
            out = torch.sum(x*disp,1)
        return out

