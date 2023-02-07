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


PYBIND11_MODULE (TORCH_EXTENSION_NAME, GANet)
{
  GANet.def ("lga_cuda_forward", &lga_cuda_forward, "lga forward (CUDA)");
  GANet.def ("lga_cuda_backward", &lga_cuda_backward, "lga backward (CUDA)");
  GANet.def ("lga3d_cuda_forward", &lga3d_cuda_forward, "lga3d forward (CUDA)");
  GANet.def ("lga3d_cuda_backward", &lga3d_cuda_backward, "lga3d backward (CUDA)");
  GANet.def ("sga_cuda_backward", &sga_cuda_backward, "sga backward (CUDA)");
  GANet.def ("sga_cuda_forward", &sga_cuda_forward, "sga forward (CUDA)");
}

