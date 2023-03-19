from torch.nn.modules.module import Module
import torch
import numpy as np
from torch.autograd import Variable
from ..functions import *

from ..functions.GANet import MyLossFunction
from ..functions.GANet import SgaFunction
from ..functions.GANet import LgaFunction
from ..functions.GANet import Lga2Function
from ..functions.GANet import Lga3Function
from ..functions.GANet import Lga3dFunction
from ..functions.GANet import Lga3d2Function
from ..functions.GANet import Lga3d3Function
from ..functions.GANet import MyLoss2Function


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
        self.maxdisp = maxdisp + 1

    def forward(self, x, y):
        assert(x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            num, channels, height, width = x.size()
            cost = x.new().resize_(num, channels * 2, self.maxdisp, height, width).zero_()
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
       self.maxdisp = maxdisp + 1
#        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(self.maxdisp)),[1,self.maxdisp,1,1])).cuda(), requires_grad=False)

    def forward(self, x):
        assert(x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            disp = Variable(torch.Tensor(np.reshape(np.array(range(self.maxdisp)),[1, self.maxdisp, 1, 1])).cuda(), requires_grad=False)
            disp = disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
            out = torch.sum(x * disp, 1)
        return out

