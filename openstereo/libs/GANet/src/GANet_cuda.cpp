//#include <torch/torch.h>
#include <torch/extension.h>
#include "GANet_kernel.h"

extern "C" int
lga_cuda_backward (at::Tensor input, at::Tensor filters,
		   at::Tensor gradOutput, at::Tensor gradInput,
		   at::Tensor gradFilters, const int radius)
{
  lga_backward (input, filters, gradOutput, gradInput, gradFilters, radius);
  return 1;
}

extern "C" int
lga_cuda_forward (at::Tensor input, at::Tensor filters, at::Tensor output,
		  const int radius)
{
  lga_forward (input, filters, output, radius);
  return 1;
}

extern "C" int
lga3d_cuda_backward (at::Tensor input, at::Tensor filters,
		     at::Tensor gradOutput, at::Tensor gradInput,
		     at::Tensor gradFilters, const int radius)
{
  lga3d_backward (input, filters, gradOutput, gradInput, gradFilters, radius);
  return 1;
}

extern "C" int
lga3d_cuda_forward (at::Tensor input, at::Tensor filters, at::Tensor output,
		    const int radius)
{
  lga3d_forward (input, filters, output, radius);
  return 1;
}

extern "C" int
sga_cuda_forward (at::Tensor input, at::Tensor guidance_down,
		  at::Tensor guidance_up, at::Tensor guidance_right,
		  at::Tensor guidance_left, at::Tensor temp_out,
		  at::Tensor output, at::Tensor mask)
{
  sga_kernel_forward (input, guidance_down, guidance_up, guidance_right,
		      guidance_left, temp_out, output, mask);
  return 1;
}

extern "C" int
sga_cuda_backward (at::Tensor input, at::Tensor guidance_down,
		   at::Tensor guidance_up, at::Tensor guidance_right,
		   at::Tensor guidance_left, at::Tensor temp_out,
		   at::Tensor mask, at::Tensor max_idx, at::Tensor gradOutput,
		   at::Tensor temp_grad, at::Tensor gradInput,
		   at::Tensor grad_down, at::Tensor grad_up,
		   at::Tensor grad_right, at::Tensor grad_left)
{
  sga_kernel_backward (input, guidance_down, guidance_up, guidance_right,
		       guidance_left, temp_out, mask, max_idx, gradOutput,
		       temp_grad, gradInput, grad_down, grad_up, grad_right,
		       grad_left);
  return 1;
}

extern "C" int
nlf_cuda_backward (at::Tensor input, at::Tensor guidance_down,
                     at::Tensor guidance_up, at::Tensor guidance_right,
                     at::Tensor guidance_left, at::Tensor output_down,
         	     at::Tensor output_up, at::Tensor output_right,
		     at::Tensor output_left, at::Tensor gradOutput,
                     at::Tensor gradInput, at::Tensor grad_down,
                     at::Tensor grad_up, at::Tensor grad_right,
                     at::Tensor grad_left)
{
    nlf_kernel_backward(input, guidance_down, guidance_up, guidance_right, guidance_left, output_down, output_up, output_right, output_left, gradOutput, gradInput, grad_down, grad_up, grad_right, grad_left);
    return 1;
}
extern "C" int
nlf_cuda_forward (at::Tensor input, at::Tensor guidance_down,
                    at::Tensor guidance_up, at::Tensor guidance_right,
                    at::Tensor guidance_left, at::Tensor output_down,
		    at::Tensor output_up, at::Tensor output_right,
 		    at::Tensor output_left)
{
    nlf_kernel_forward(input, guidance_down, guidance_up, guidance_right, guidance_left, output_down, output_up, output_right, output_left);
    return 1;
}

extern "C" int
nlf_down_cuda_forward (at::Tensor input, at::Tensor guidance_down, at::Tensor output_down){
    nlf_down_kernel_forward (input, guidance_down, output_down);
    return 1;

}
extern "C" int
nlf_down_cuda_backward (at::Tensor input, at::Tensor guidance_down, at::Tensor output_down, at::Tensor gradOutput,
                     at::Tensor gradInput, at::Tensor grad_down){
    nlf_down_kernel_backward (input, guidance_down, output_down, gradOutput, gradInput, grad_down);
    return 1;
}
extern "C" int
nlf_up_cuda_forward (at::Tensor input, at::Tensor guidance_up, at::Tensor output_up){
    nlf_up_kernel_forward (input, guidance_up, output_up);
    return 1;
}
extern "C" int
nlf_up_cuda_backward (at::Tensor input, at::Tensor guidance_up, at::Tensor output_up, at::Tensor gradOutput,
                     at::Tensor gradInput, at::Tensor grad_up){
    nlf_up_kernel_backward (input, guidance_up, output_up, gradOutput, gradInput, grad_up);
    return 1;
}
extern "C" int
nlf_left_cuda_forward (at::Tensor input, at::Tensor guidance_left, at::Tensor output_left){
    nlf_left_kernel_forward (input, guidance_left, output_left);
    return 1;
}
extern "C" int
nlf_left_cuda_backward (at::Tensor input, at::Tensor guidance_left, at::Tensor output_left, at::Tensor gradOutput,
                     at::Tensor gradInput, at::Tensor grad_left){
    nlf_left_kernel_backward (input, guidance_left, output_left, gradOutput, gradInput, grad_left);
    return 1;
}
extern "C" int
nlf_right_cuda_forward (at::Tensor input, at::Tensor guidance_right, at::Tensor output_right){
    nlf_right_kernel_forward (input, guidance_right, output_right);
    return 1;
}
extern "C" int
nlf_right_cuda_backward (at::Tensor input, at::Tensor guidance_right, at::Tensor output_right, at::Tensor gradOutput,
                     at::Tensor gradInput, at::Tensor grad_right){
    nlf_right_kernel_backward (input, guidance_right, output_right, gradOutput, gradInput, grad_right);
    return 1;
}
/*
extern "C" int
lgf_cuda_forward (at::Tensor input, at::Tensor filters, at::Tensor output,
	     const int radius){
    lgf_kernel_forward (input, filters, output, radius);
    return 1;
}
extern "C" int
lgf_cuda_backward (at::Tensor input, at::Tensor filters, at::Tensor gradOutput,
	      at::Tensor gradInput, at::Tensor gradFilters, const int radius){
    lgf_kernel_backward (input, filters, gradOutput, gradInput, gradFilters, radius);
    return 1;
}*/
/*
extern "C" int
sgf3d_down_cuda_forward (at::Tensor input, at::Tensor guidance_down, at::Tensor output_down){
    sgf3d_down_kernel_forward (input, guidance_down, output_down);
    return 1;

}
extern "C" int
sgf3d_down_cuda_backward (at::Tensor input, at::Tensor guidance_down, at::Tensor output_down, at::Tensor gradOutput,
                     at::Tensor gradInput, at::Tensor grad_down){
    sgf3d_down_kernel_backward (input, guidance_down, output_down, gradOutput, gradInput, grad_down);
    return 1;
}
extern "C" int
sgf3d_up_cuda_forward (at::Tensor input, at::Tensor guidance_up, at::Tensor output_up){
    sgf3d_up_kernel_forward (input, guidance_up, output_up);
    return 1;
}
extern "C" int
sgf3d_up_cuda_backward (at::Tensor input, at::Tensor guidance_up, at::Tensor output_up, at::Tensor gradOutput,
                     at::Tensor gradInput, at::Tensor grad_up){
    sgf3d_up_kernel_backward (input, guidance_up, output_up, gradOutput, gradInput, grad_up);
    return 1;
}
extern "C" int
sgf3d_left_cuda_forward (at::Tensor input, at::Tensor guidance_left, at::Tensor output_left){
    sgf3d_left_kernel_forward (input, guidance_left, output_left);
    return 1;
}
extern "C" int
sgf3d_left_cuda_backward (at::Tensor input, at::Tensor guidance_left, at::Tensor output_left, at::Tensor gradOutput,
                     at::Tensor gradInput, at::Tensor grad_left){
    sgf3d_left_kernel_backward (input, guidance_left, output_left, gradOutput, gradInput, grad_left);
    return 1;
}
extern "C" int
sgf3d_right_cuda_forward (at::Tensor input, at::Tensor guidance_right, at::Tensor output_right){
    sgf3d_right_kernel_forward (input, guidance_right, output_right);
    return 1;
}
extern "C" int
sgf3d_right_cuda_backward (at::Tensor input, at::Tensor guidance_right, at::Tensor output_right, at::Tensor gradOutput,
                     at::Tensor gradInput, at::Tensor grad_right){
    sgf3d_right_kernel_backward (input, guidance_right, output_right, gradOutput, gradInput, grad_right);
    return 1;
}
*/
PYBIND11_MODULE (TORCH_EXTENSION_NAME, GANet)
{
  GANet.def ("lga_cuda_forward", &lga_cuda_forward, "lga forward (CUDA)");
  GANet.def ("lga_cuda_backward", &lga_cuda_backward, "lga backward (CUDA)");
  GANet.def ("lga3d_cuda_forward", &lga3d_cuda_forward, "lga3d forward (CUDA)");
  GANet.def ("lga3d_cuda_backward", &lga3d_cuda_backward, "lga3d backward (CUDA)");
  GANet.def ("sga_cuda_backward", &sga_cuda_backward, "sga backward (CUDA)");
  GANet.def ("sga_cuda_forward", &sga_cuda_forward, "sga forward (CUDA)");
  GANet.def ("nlf_cuda_backward", &nlf_cuda_backward, "sgf backward (CUDA)");
  GANet.def ("nlf_cuda_forward", &nlf_cuda_forward, "sgf forward (CUDA)");
  GANet.def ("nlf_down_cuda_forward", &nlf_down_cuda_forward, "sgf down forward (CUDA)");
  GANet.def ("nlf_down_cuda_backward", &nlf_down_cuda_backward, "sgf down backward (CUDA)");
  GANet.def ("nlf_up_cuda_forward", &nlf_up_cuda_forward, "sgf up forward (CUDA)");
  GANet.def ("nlf_up_cuda_backward", &nlf_up_cuda_backward, "sgf up backward (CUDA)");
  GANet.def ("nlf_right_cuda_forward", &nlf_right_cuda_forward, "sgf right forward (CUDA)");
  GANet.def ("nlf_right_cuda_backward", &nlf_right_cuda_backward, "sgf right backward (CUDA)");
  GANet.def ("nlf_left_cuda_forward", &nlf_left_cuda_forward, "sgf left forward (CUDA)");
  GANet.def ("nlf_left_cuda_backward", &nlf_left_cuda_backward, "sgf left backward (CUDA)");
//  GANet.def ("lgf_cuda_forward", &lgf_cuda_forward, "lgf forward (CUDA)");
//  GANet.def ("lgf_cuda_backward", &lgf_cuda_backward, "lgf backward (CUDA)");
/*  GANet.def ("sgf3d_down_cuda_forward", &sgf3d_down_cuda_forward, "sgf3d down forward (CUDA)");
  GANet.def ("sgf3d_down_cuda_backward", &sgf3d_down_cuda_backward, "sgf3d down backward (CUDA)");
  GANet.def ("sgf3d_up_cuda_forward", &sgf3d_up_cuda_forward, "sgf3d up forward (CUDA)");
  GANet.def ("sgf3d_up_cuda_backward", &sgf3d_up_cuda_backward, "sgf3d up backward (CUDA)");
  GANet.def ("sgf3d_right_cuda_forward", &sgf3d_right_cuda_forward, "sgf3d right forward (CUDA)");
  GANet.def ("sgf3d_right_cuda_backward", &sgf3d_right_cuda_backward, "sgf3d right backward (CUDA)");
  GANet.def ("sgf3d_left_cuda_forward", &sgf3d_left_cuda_forward, "sgf3d left forward (CUDA)");
  GANet.def ("sgf3d_left_cuda_backward", &sgf3d_left_cuda_backward, "sgf3d left backward (CUDA)");
*/
}

