int lga_cuda_backward (at::Tensor input, at::Tensor filters,
		       at::Tensor gradOutput, at::Tensor gradInput,
		       at::Tensor gradFilters, const int radius);
int lga_cuda_forward (at::Tensor input, at::Tensor filters, at::Tensor output,
		      const int radius);
int lga3d_cuda_backward (at::Tensor input, at::Tensor filters,
			 at::Tensor gradOutput, at::Tensor gradInput,
			 at::Tensor gradFilters, const int radius);
int lga3d_cuda_forward (at::Tensor input, at::Tensor filters,
			at::Tensor output, const int radius);
int sga_cuda_forward (at::Tensor input, at::Tensor guidance_down,
		      at::Tensor guidance_up, at::Tensor guidance_right,
		      at::Tensor guidance_left, at::Tensor temp_out,
		      at::Tensor output, at::Tensor mask);
int sga_cuda_backward (at::Tensor input, at::Tensor guidance_down,
		       at::Tensor guidance_up, at::Tensor guidance_right,
		       at::Tensor guidance_left, at::Tensor temp_out,
		       at::Tensor mask, at::Tensor max_idx,
		       at::Tensor gradOutput, at::Tensor temp_grad,
		       at::Tensor gradInput, at::Tensor grad_down,
		       at::Tensor grad_up, at::Tensor grad_right,
		       at::Tensor grad_left);

