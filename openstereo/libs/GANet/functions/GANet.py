import torch
from torch.autograd import Function
from ..build.lib import GANet
from torch.autograd import Variable
#import GANet
class NlfDownFunction(Function):
    @staticmethod
    def forward(ctx, input, g0):
        assert(input.is_contiguous() == True and g0.is_contiguous() == True)
        with torch.cuda.device_of(input):
            output_down = input.new().resize_(input.size()).zero_()
            GANet.nlf_down_cuda_forward(input, g0, output_down)
            output_down = output_down.contiguous()
        ctx.save_for_backward(input, g0, output_down)
        return output_down
    @staticmethod
    def backward(ctx, gradOutput):
        input, g0, output_down = ctx.saved_tensors
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):

            gradInput = gradOutput.new().resize_(input.size()).zero_()
            grad0 = gradOutput.new().resize_(g0.size()).zero_()
            GANet.nlf_down_cuda_backward(input, g0, output_down, gradOutput, gradInput, grad0)
            gradInput = gradInput.contiguous()
            grad0 = grad0.contiguous()
        return gradInput, grad0
class NlfUpFunction(Function):
    @staticmethod
    def forward(ctx, input, g1):
        assert(input.is_contiguous() == True and g1.is_contiguous() == True)
        with torch.cuda.device_of(input):
            output_up = input.new().resize_(input.size()).zero_()
            GANet.nlf_up_cuda_forward(input, g1, output_up)
            output_up = output_up.contiguous()
        ctx.save_for_backward(input, g1, output_up)
        return output_up
    @staticmethod
    def backward(ctx, gradOutput):
        input, g1, output_up = ctx.saved_tensors
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
            gradInput = gradOutput.new().resize_(input.size()).zero_()
            grad1 = gradOutput.new().resize_(g1.size()).zero_()
            GANet.nlf_up_cuda_backward(input, g1, output_up, gradOutput, gradInput, grad1)
            gradInput = gradInput.contiguous()
            grad1 = grad1.contiguous()
        return gradInput, grad1
class NlfRightFunction(Function):
    @staticmethod
    def forward(ctx, input, g2):
        assert(input.is_contiguous() == True and g2.is_contiguous() == True)
        with torch.cuda.device_of(input):
#            num, channels, height, width = input.size()
#            output_right = input.new().resize_(num, channels, height, width).zero_()
            output_right = input.new().resize_(input.size()).zero_()
            GANet.nlf_right_cuda_forward(input, g2, output_right)
            output_right = output_right.contiguous()
        ctx.save_for_backward(input, g2, output_right)
        return output_right
    @staticmethod
    def backward(ctx, gradOutput):
        input, g2, output_right = ctx.saved_tensors
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
#            num, channels, height, width = input.size()
#            _, fsize, _, _ = g2.size()
#            gradInput = gradOutput.new().resize_(num, channels, height, width).zero_()
#            grad2 = gradOutput.new().resize_(num, fsize, height, width).zero_()
            gradInput = gradOutput.new().resize_(input.size()).zero_()
            grad2 = gradOutput.new().resize_(g2.size()).zero_()
            GANet.nlf_right_cuda_backward(input, g2, output_right, gradOutput, gradInput, grad2)
            gradInput = gradInput.contiguous()
            grad2 = grad2.contiguous()
        return gradInput, grad2
class NlfLeftFunction(Function):
    @staticmethod
    def forward(ctx, input, g3):

        assert(input.is_contiguous() == True and g3.is_contiguous() == True)
        with torch.cuda.device_of(input):
#            num, channels, height, width = input.size()
#            output_left = input.new().resize_(num, channels, height, width).zero_()
            output_left = input.new().resize_(input.size()).zero_()
            GANet.nlf_left_cuda_forward(input, g3, output_left)
            output_left = output_left.contiguous()
        ctx.save_for_backward(input, g3, output_left)
        return output_left
    @staticmethod
    def backward(ctx, gradOutput):
        input, g3, output_left = ctx.saved_tensors
        gradOutput = gradOutput.contiguous()
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
#            num, channels, height, width = input.size()
#            _, fsize, _, _ = g3.size()
#            gradInput = gradOutput.new().resize_(num, channels, height, width).zero_()
#            grad3 = gradOutput.new().resize_(num, fsize, height, width).zero_()
            gradInput = gradOutput.new().resize_(input.size()).zero_()
            grad3 = gradOutput.new().resize_(g3.size()).zero_()
            GANet.nlf_left_cuda_backward(input, g3, output_left, gradOutput, gradInput, grad3)
            gradInput = gradInput.contiguous()
            grad3 = grad3.contiguous()
        return gradInput, grad3
		
class NlfFunction(Function):
    @staticmethod
    def forward(ctx, input, g0, g1, g2, g3):
        assert(input.is_contiguous() == True and g0.is_contiguous() == True and g1.is_contiguous() == True and g2.is_contiguous() == True and g3.is_contiguous() == True)
        with torch.cuda.device_of(input):
            num, channels, height, width = input.size()
            output_down = input.new().resize_(num, channels, height, width).zero_()
            output_up = input.new().resize_(num, channels, height, width).zero_()
            output_right = input.new().resize_(num, channels, height, width).zero_()
            output_left = input.new().resize_(num, channels, height, width).zero_()
            GANet.nlf_cuda_forward(input, g0, g1, g2, g3, output_down, output_up, output_right, output_left)
 #           GANet.sga_cuda_forward(input, filters, output, radius)
            
            output_down = output_down.contiguous()
            output_up = output_up.contiguous()
            output_right = output_right.contiguous()
            output_left = output_left.contiguous()
        ctx.save_for_backward(input, g0, g1, g2, g3, output_down, output_up, output_right, output_left)
#        print(output_left.size())
        return output_left
    @staticmethod
    def backward(ctx, gradOutput):
        input, g0, g1, g2, g3, output_down, output_up, output_right, output_left = ctx.saved_tensors
#        print temp_out.size()
#        print mask.size()
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
            num, channels, height, width = input.size()
            _, _, fsize, _, _ = g0.size()
#            print fsize            
            gradInput = gradOutput.new().resize_(num, channels, height, width).zero_()
            grad0 = gradOutput.new().resize_(num, channels, fsize, height, width).zero_()
            grad1 = gradOutput.new().resize_(num, channels, fsize, height, width).zero_()
            grad2 = gradOutput.new().resize_(num, channels, fsize, height, width).zero_()
            grad3 = gradOutput.new().resize_(num, channels, fsize, height, width).zero_()

            GANet.nlf_cuda_backward(input, g0, g1, g2, g3, output_down, output_up, output_right, output_left, gradOutput, gradInput, grad0, grad1, grad2, grad3)
#            GANet.lga_cuda_backward(input, filters, gradOutput, gradInput, gradFilters, radius)
            gradInput = gradInput.contiguous()
            grad0 = grad0.contiguous()
            grad1 = grad1.contiguous()
            grad2 = grad2.contiguous()
            grad3 = grad3.contiguous()
        return gradInput, grad0, grad1, grad2, grad3

class SgaFunction(Function):
    @staticmethod
    def forward(ctx, input, g0, g1, g2, g3):
        assert(input.is_contiguous() == True and g0.is_contiguous() == True and g1.is_contiguous() == True and g2.is_contiguous() == True and g3.is_contiguous() == True)
        with torch.cuda.device_of(input):
            num, channels, depth, height, width = input.size()
            output = input.new().resize_(num, channels, depth, height, width).zero_()
            temp_out = input.new().resize_(num, channels, depth, height, width).zero_()
            mask = input.new().resize_(num, channels, depth, height, width).zero_()
            GANet.sga_cuda_forward(input, g0, g1, g2, g3, temp_out, output, mask)
 #           GANet.sga_cuda_forward(input, filters, output, radius)
            
            output = output.contiguous()
        ctx.save_for_backward(input, g0, g1, g2, g3, temp_out, mask)
        return output
    @staticmethod
    def backward(ctx, gradOutput):
        input, g0, g1, g2, g3, temp_out, mask = ctx.saved_tensors
#        print temp_out.size()
#        print mask.size()
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
            num, channels, depth, height, width = input.size()
#            _, _, fsize, _, _ = g0.size()
#            print fsize            
            gradInput = gradOutput.new().resize_(num, channels, depth, height, width).zero_()
            grad0 = gradOutput.new().resize_(g0.size()).zero_()
            grad1 = gradOutput.new().resize_(g1.size()).zero_()
            grad2 = gradOutput.new().resize_(g2.size()).zero_()
            grad3 = gradOutput.new().resize_(g3.size()).zero_()
            temp_grad = gradOutput.new().resize_(num, channels, depth, height, width).zero_()     
            max_idx = gradOutput.new().resize_(num, channels, height, width).zero_()    

            GANet.sga_cuda_backward(input, g0, g1, g2, g3, temp_out, mask, max_idx, gradOutput, temp_grad, gradInput, grad0, grad1, grad2, grad3)
#            GANet.lga_cuda_backward(input, filters, gradOutput, gradInput, gradFilters, radius)
            gradInput = gradInput.contiguous()
            grad0 = grad0.contiguous()
            grad1 = grad1.contiguous()
            grad2 = grad2.contiguous()
            grad3 = grad3.contiguous()
        return gradInput, grad0, grad1, grad2, grad3
		
		
class Lga3d3Function(Function):
    @staticmethod      
    def forward(ctx, input, filters, radius=1):
        ctx.radius = radius
        assert(input.is_contiguous() == True and filters.is_contiguous() == True)
        with torch.cuda.device_of(input):
            num, channels, depth, height, width = input.size()
            temp_out1 = input.new().resize_(num, channels, depth, height, width).zero_()
            temp_out2 = input.new().resize_(num, channels, depth, height, width).zero_()
            output = input.new().resize_(num, channels, depth, height, width).zero_()
            GANet.lga3d_cuda_forward(input, filters, temp_out1, radius)
            GANet.lga3d_cuda_forward(temp_out1, filters, temp_out2, radius)
            GANet.lga3d_cuda_forward(temp_out2, filters, output, radius)
            output = output.contiguous()
        ctx.save_for_backward(input, filters, temp_out1, temp_out2)
        return output
    @staticmethod
    def backward(ctx, gradOutput):
        input, filters, temp_out1, temp_out2 = ctx.saved_tensors
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
            num, channels, depth, height, width = input.size()
            _, _, fsize, _, _ = filters.size()
#            gradInput = gradOutput.new().resize_(num, channels, height, width).zero_()
            gradFilters = gradOutput.new().resize_(num, channels, fsize, height, width).zero_()
            GANet.lga3d_cuda_backward(temp_out2, filters, gradOutput, temp_out2, gradFilters, ctx.radius)
            GANet.lga3d_cuda_backward(temp_out1, filters, temp_out2, temp_out1, gradFilters, ctx.radius)
#            temp_out[...] = 0
            GANet.lga3d_cuda_backward(input, filters, temp_out1, temp_out2, gradFilters, ctx.radius)
#            temp_out[...] = gradOutput[...]
            temp_out2 = temp_out2.contiguous()
            gradFilters = gradFilters.contiguous()
        return temp_out2, gradFilters, None
class Lga3d2Function(Function):
    @staticmethod      
    def forward(ctx, input, filters, radius=1):
        ctx.radius = radius
        assert(input.is_contiguous() == True and filters.is_contiguous() == True)
        with torch.cuda.device_of(input):
            num, channels, depth, height, width = input.size()
            temp_out = input.new().resize_(num, channels, depth, height, width).zero_()
            output = input.new().resize_(num, channels, depth, height, width).zero_()
            GANet.lga3d_cuda_forward(input, filters, temp_out, radius)
            GANet.lga3d_cuda_forward(temp_out, filters, output, radius)
            output = output.contiguous()
        ctx.save_for_backward(input, filters, temp_out)
        return output
    @staticmethod
    def backward(ctx, gradOutput):
        input, filters, temp_out = ctx.saved_tensors
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
            num, channels, depth, height, width = input.size()
            _, _, fsize, _, _ = filters.size()
#            gradInput = gradOutput.new().resize_(num, channels, height, width).zero_()
            gradFilters = gradOutput.new().resize_(num, channels, fsize, height, width).zero_()
            GANet.lga3d_cuda_backward(temp_out, filters, gradOutput, temp_out, gradFilters, ctx.radius)
#            temp_out[...] = 0
            GANet.lga3d_cuda_backward(input, filters, temp_out, gradOutput, gradFilters, ctx.radius)
            temp_out[...] = gradOutput[...]
            temp_out = temp_out.contiguous()
            gradFilters = gradFilters.contiguous()
        return temp_out, gradFilters, None

class Lga3dFunction(Function):
    @staticmethod
    def forward(ctx, input, filters, radius=1):
        ctx.radius = radius
        ctx.save_for_backward(input, filters)
        assert(input.is_contiguous() == True and filters.is_contiguous() == True)
        with torch.cuda.device_of(input):
            num, channels, depth, height, width = input.size()
            output = input.new().resize_(num, channels, depth, height, width).zero_()
            GANet.lga3d_cuda_forward(input, filters, output, radius)
            output = output.contiguous()
        return output
    @staticmethod
    def backward(ctx, gradOutput):
        input, filters = ctx.saved_tensors
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
            num, channels, depth, height, width = input.size()
            _, _, fsize, _, _ = filters.size()
            gradInput = gradOutput.new().resize_(num, channels, depth, height, width).zero_()
            gradFilters = gradOutput.new().resize_(num, channels, fsize, height, width).zero_()
            GANet.lga3d_cuda_backward(input, filters, gradOutput, gradInput, gradFilters, ctx.radius)
            gradInput = gradInput.contiguous()
            gradFilters = gradFilters.contiguous()
        return gradInput, gradFilters, None

class Lga3Function(Function):
    @staticmethod     
    def forward(ctx, input, filters, radius=1):
        ctx.radius = radius
        assert(input.is_contiguous() == True and filters.is_contiguous() == True)
        with torch.cuda.device_of(input):
            num, channels, height, width = input.size()
            temp_out1 = input.new().resize_(num, channels, height, width).zero_()
            temp_out2 = input.new().resize_(num, channels, height, width).zero_()
            output = input.new().resize_(num, channels, height, width).zero_()
            GANet.lga_cuda_forward(input, filters, temp_out1, radius)
            GANet.lga_cuda_forward(temp_out1, filters, temp_out2, radius)
            GANet.lga_cuda_forward(temp_out2, filters, output, radius)
            output = output.contiguous()
        ctx.save_for_backward(input, fitlers, temp_out1, temp_out2)
        return output
    @staticmethod
    def backward(ctx, gradOutput):
        input, filters, temp_out1, temp_out2 = ctx.saved_tensors
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
            num, channels, height, width = input.size()
            _, fsize, _, _ = filters.size()
#            gradInput = gradOutput.new().resize_(num, channels, height, width).zero_()
            gradFilters = gradOutput.new().resize_(num, fsize, height, width).zero_()
            GANet.lga_cuda_backward(temp_out2, filters, gradOutput, temp_out2, gradFilters, ctx.radius)
            GANet.lga_cuda_backward(temp_out1, filters, temp_out2, temp_out1, gradFilters, ctx.radius)
#            temp_out[...] = 0
            GANet.lga_cuda_backward(input, filters, temp_out1, temp_out2, gradFilters, ctx.radius)
#            temp_out[...] = gradOutput[...]
            temp_out2 = temp_out2.contiguous()
            gradFilters = gradFilters.contiguous()
        return temp_out2, gradFilters, None
class Lga2Function(Function):   
    @staticmethod  
    def forward(ctx, input, filters, radius=1):
        ctx.radius = radius
        assert(input.is_contiguous() == True and filters.is_contiguous() == True)
        with torch.cuda.device_of(input):
            num, channels, height, width = input.size()
            temp_out = input.new().resize_(num, channels, height, width).zero_()
            output = input.new().resize_(num, channels, height, width).zero_()
            GANet.lga_cuda_forward(input, filters, temp_out, radius)
            GANet.lga_cuda_forward(temp_out, filters, output, radius)
            output = output.contiguous()
        ctx.save_for_backward(input, filters, temp_out)
        return output
    @staticmethod
    def backward(ctx, gradOutput):
        input, filters, temp_out = ctx.saved_tensors
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
            num, channels, height, width = input.size()
            _, fsize, _, _ = filters.size()
#            gradInput = gradOutput.new().resize_(num, channels, height, width).zero_()
            gradFilters = gradOutput.new().resize_(num, fsize, height, width).zero_()
            GANet.lga_cuda_backward(temp_out, filters, gradOutput, temp_out, gradFilters, ctx.radius)
#            temp_out[...] = 0
            GANet.lga_cuda_backward(input, filters, temp_out, gradOutput, gradFilters, ctx.radius)
            temp_out[...] = gradOutput[...]
            temp_out = temp_out.contiguous()
            gradFilters = gradFilters.contiguous()
        return temp_out, gradFilters, None

class Lgf2Function(Function):
    @staticmethod       
    def forward(ctx, input, filters, radius=2):
        ctx.radius = radius
        assert(input.is_contiguous() == True and filters.is_contiguous() == True)
        with torch.cuda.device_of(input):
#            num, channels, depth, height, width = input.size()
#            temp_out = input.new().resize_(num, channels, depth, height, width).zero_()
#            output = input.new().resize_(num, channels, depth, height, width).zero_()
            temp_out = input.new().resize_(input.size()).zero_()
            output = input.new().resize_(input.size()).zero_()
            GANet.lgf_cuda_forward(input, filters, temp_out, radius)
            GANet.lgf_cuda_forward(temp_out, filters, output, radius)
            output = output.contiguous()
        ctx.save_for_backward(input, filters, temp_out)
        return output
    @staticmethod
    def backward(ctx, gradOutput):
        input, filters, temp_out = ctx.saved_tensors
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
#            num, channels, depth, height, width = input.size()
#            _, fsize, _, _ = filters.size()
#            gradInput = gradOutput.new().resize_(num, channels, height, width).zero_()
#            gradFilters = gradOutput.new().resize_(num, fsize, height, width).zero_()
            gradFilters = gradOutput.new().resize_(filters.size()).zero_()
            GANet.lgf_cuda_backward(temp_out, filters, gradOutput, temp_out, gradFilters, ctx.radius)
#            temp_out[...] = 0
            GANet.lgf_cuda_backward(input, filters, temp_out, gradOutput, gradFilters, ctx.radius)
            temp_out[...] = gradOutput[...]
            temp_out = temp_out.contiguous()
            gradFilters = gradFilters.contiguous()
        return temp_out, gradFilters, None

class LgaFunction(Function):
    @staticmethod
    def forward(ctx, input, filters):
        ctx.radius = radius
        assert(input.is_contiguous() == True and filters.is_contiguous() == True)
        with torch.cuda.device_of(input):
            num, channels, height, width = input.size()
            output = input.new().resize_(num, channels, height, width).zero_()
            GANet.lga_cuda_forward(input, filters, output, radius)
            output = output.contiguous()
        ctx.save_for_backward(input, filters)
        return output
    @staticmethod
    def backward(ctx, gradOutput):
        input, filters = ctx.saved_tensors
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
            num, channels, height, width = input.size()
            _, fsize, _, _ = filters.size()
            gradInput = gradOutput.new().resize_(num, channels, height, width).zero_()
            gradFilters = gradOutput.new().resize_(num, fsize, height, width).zero_()
            GANet.lga_cuda_backward(input, filters, gradOutput, gradInput, gradFilters, ctx.radius)
            gradInput = gradInput.contiguous()
            gradFilters = gradFilters.contiguous()
        return gradInput, gradFilters, None
class MyLoss2Function(Function):
    @staticmethod
    def forward(ctx, input1, input2, thresh=1, alpha=2):
        ctx.thresh = thresh
        ctx.alpha = alpha
        diff = input1 - input2
        temp=torch.abs(diff)
        temp[temp < thresh] = temp[temp < thresh] ** 2 / thresh
        tag = (temp <= thresh + alpha) & (temp >= thresh)
        temp[tag]=temp[tag] * 2 - (temp[tag] - thresh) ** 2 /(2.0 * alpha) - thresh
        temp[temp > thresh + alpha] += (alpha / 2.0)
        ctx.save_for_backward(diff)
        return torch.mean(temp)
    @staticmethod
    def backward(ctx, gradOutput):
        diff, = ctx.saved_tensors
        scale = torch.abs(diff)
        scale[scale > ctx.thresh + ctx.alpha] = 1
        tag = (scale <= ctx.thresh + ctx.alpha) & (scale >= ctx.thresh)
        scale[tag] = 2 - (scale[tag] - ctx.thresh) / ctx.alpha
        tag = scale < ctx.thresh
        scale[tag] = 2*scale[tag] / ctx.thresh
        diff[diff > 0] = 1.0
        diff[diff < 0] = -1.0
        diff = diff * scale * gradOutput / scale.numel()
        return diff, Variable(torch.Tensor([0])), None, None

class MyLossFunction(Function):

    @staticmethod
    def forward(ctx, input1, input2, upper_thresh=5, lower_thresh=1):
        ctx.upper_thresh = upper_thresh
        ctx.lower_thresh = lower_thresh
        diff = input1 - input2
        ctx.save_for_backward(diff)
        return torch.mean(torch.abs(diff))
    @staticmethod
    def backward(ctx, gradOutput):
        diff, = ctx.saved_tensors
        scale = torch.abs(diff)
        scale[scale > ctx.upper_thresh] = 1
        tag = (scale <= ctx.upper_thresh) & (scale >= ctx.lower_thresh)
        scale[tag] = 2 - torch.abs(scale[tag]-(ctx.upper_thresh + ctx.lower_thresh)/2.)/2.
        diff[diff > 0] = 1
        diff[diff < 0] = -1
        diff = diff * scale * gradOutput
        return diff, Variable(torch.Tensor([0])), None, None


