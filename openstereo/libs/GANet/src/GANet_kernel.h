
#include <torch/extension.h>

#ifdef __cplusplus
    extern "C" {
#endif
void nlf_kernel_backward (at::Tensor input, at::Tensor guidance_down,
                     at::Tensor guidance_up, at::Tensor guidance_right,
                     at::Tensor guidance_left, at::Tensor output_down,
         	     at::Tensor output_up, at::Tensor output_right,
		     at::Tensor output_left, at::Tensor gradOutput,
                     at::Tensor gradInput, at::Tensor grad_down,
                     at::Tensor grad_up, at::Tensor grad_right,
                     at::Tensor grad_left);
void nlf_kernel_forward (at::Tensor input, at::Tensor guidance_down,
                    at::Tensor guidance_up, at::Tensor guidance_right,
                    at::Tensor guidance_left, at::Tensor output_down,
		    at::Tensor output_up, at::Tensor output_right,
 		    at::Tensor output_left);


void sga_kernel_forward (at::Tensor input, at::Tensor guidance_down,
			 at::Tensor guidance_up, at::Tensor guidance_right,
			 at::Tensor guidance_left, at::Tensor temp_out,
			 at::Tensor output, at::Tensor mask);
void sga_kernel_backward (at::Tensor input, at::Tensor guidance_down,
			  at::Tensor guidance_up, at::Tensor guidance_right,
			  at::Tensor guidance_left, at::Tensor temp_out,
			  at::Tensor mask, at::Tensor max_idx,
			  at::Tensor gradOutput, at::Tensor temp_grad,
			  at::Tensor gradInput, at::Tensor grad_down,
			  at::Tensor grad_up, at::Tensor grad_right,
			  at::Tensor grad_left);

void lga_backward (at::Tensor input, at::Tensor filters,
		   at::Tensor gradOutput, at::Tensor gradInput,
		   at::Tensor gradFilters, const int radius);
void lga_forward (at::Tensor input, at::Tensor filters, at::Tensor output,
		  const int radius);

void lga3d_backward (at::Tensor input, at::Tensor filters,
		     at::Tensor gradOutput, at::Tensor gradInput,
		     at::Tensor gradFilters, const int radius);
void lga3d_forward (at::Tensor input, at::Tensor filters, at::Tensor output,
		    const int radius);
void nlf_down_kernel_forward (at::Tensor input, at::Tensor guidance_down, at::Tensor output_down);
void nlf_down_kernel_backward (at::Tensor input, at::Tensor guidance_down, at::Tensor output_down, at::Tensor gradOutput,
                     at::Tensor gradInput, at::Tensor grad_down);
void nlf_up_kernel_forward (at::Tensor input, at::Tensor guidance_up, at::Tensor output_up);
void nlf_up_kernel_backward (at::Tensor input, at::Tensor guidance_up, at::Tensor output_up, at::Tensor gradOutput,
                     at::Tensor gradInput, at::Tensor grad_up);

void nlf_right_kernel_forward (at::Tensor input, at::Tensor guidance_right, at::Tensor output_right);
void nlf_right_kernel_backward (at::Tensor input, at::Tensor guidance_right, at::Tensor output_right, at::Tensor gradOutput,
                     at::Tensor gradInput, at::Tensor grad_right);

void nlf_left_kernel_forward (at::Tensor input, at::Tensor guidance_left, at::Tensor output_left);
void nlf_left_kernel_backward (at::Tensor input, at::Tensor guidance_left, at::Tensor output_left, at::Tensor gradOutput,
                     at::Tensor gradInput, at::Tensor grad_left);

/*
void sgf3d_down_kernel_forward (at::Tensor input, at::Tensor guidance_down, at::Tensor output_down);
void sgf3d_down_kernel_backward (at::Tensor input, at::Tensor guidance_down, at::Tensor output_down, at::Tensor gradOutput,
                     at::Tensor gradInput, at::Tensor grad_down);
void sgf3d_up_kernel_forward (at::Tensor input, at::Tensor guidance_up, at::Tensor output_up);
void sgf3d_up_kernel_backward (at::Tensor input, at::Tensor guidance_up, at::Tensor output_up, at::Tensor gradOutput,
                     at::Tensor gradInput, at::Tensor grad_up);

void sgf3d_right_kernel_forward (at::Tensor input, at::Tensor guidance_right, at::Tensor output_right);
void sgf3d_right_kernel_backward (at::Tensor input, at::Tensor guidance_right, at::Tensor output_right, at::Tensor gradOutput,
                     at::Tensor gradInput, at::Tensor grad_right);

void sgf3d_left_kernel_forward (at::Tensor input, at::Tensor guidance_left, at::Tensor output_left);
void sgf3d_left_kernel_backward (at::Tensor input, at::Tensor guidance_left, at::Tensor output_left, at::Tensor gradOutput,
                     at::Tensor gradInput, at::Tensor grad_left);
*/
#ifdef __cplusplus
    }
#endif
