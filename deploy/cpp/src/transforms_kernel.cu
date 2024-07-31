#include "transforms_kernel.h"

// =========================== OpenCV CPU Transformation ========================== //
// ================================ PadImageKernel ================================ //

__global__ void PadImageKernel(float* input, float* output, int input_rows, int input_cols, int input_channels, int output_rows, int output_cols, int pad_top, int pad_left, float fill_value[3], int border_type) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < output_cols && y < output_rows) {
        int input_x = x - pad_left;
        int input_y = y - pad_top;

        float *output_ptr = &output[(y * output_cols + x) * input_channels];

        if (input_x >= 0 && input_x < input_cols && input_y >= 0 && input_y < input_rows) {
            float *input_ptr = &input[(input_y * input_cols + input_x) * input_channels];
            for (int c = 0; c < input_channels; ++c) {
                output_ptr[c] = input_ptr[c];
            }
        } else {
            if (border_type == 0) {  // constant fill
                for (int c = 0; c < input_channels; ++c) {
                    output_ptr[c] = fill_value[c];
                }
            } else if (border_type == 1) {  // replicate fill
                int clamped_x = fmaxf(0, fminf(input_x, input_cols - 1));
                int clamped_y = fmaxf(0, fminf(input_y, input_rows - 1));
                float *input_ptr = &input[(clamped_y * input_cols + clamped_x) * input_channels];
                for (int c = 0; c < input_channels; ++c) {
                    output_ptr[c] = input_ptr[c];
                }
            }
        }
    }
}

void PadImageCUDA(cv::Mat& image, int target_height, int target_width, float fill_value[3], int border_type) {
    int input_rows = image.rows;
    int input_cols = image.cols;
    int input_channels = image.channels();
    int pad_top = target_height - input_rows;
    int pad_left = target_width - input_cols;

    size_t input_size = input_rows * input_cols * input_channels * sizeof(float);
    size_t output_size = target_height * target_width * input_channels * sizeof(float);

    float* d_input;
    float* d_output;

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);

    cudaMemcpy(d_input, image.ptr<float>(), input_size, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, output_size); // flash buffer

    dim3 blockDim(16, 16);
    dim3 gridDim((target_width + blockDim.x - 1) / blockDim.x, (target_height + blockDim.y - 1) / blockDim.y);

    PadImageKernel<<<gridDim, blockDim>>>(d_input, d_output, input_rows, input_cols, input_channels, target_height, target_width, pad_top, pad_left, fill_value, border_type);

    cudaFree(d_input);

    image.create(target_height, target_width, CV_32FC3); // output image
    cudaMemcpy(image.ptr<float>(), d_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(d_output);
}

// =========================== Fused GPU Transformation =========================== //
// =================== FusedRightTopPadTransposeNormalizeKernel =================== //

__global__ void FusedRightTopPadTransposeNormalizeKernel(const float* __restrict__ input_data1, const float* __restrict__ input_data2, float* __restrict__ output_data1, float* __restrict__ output_data2, FusedParams params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = params.target_height * params.target_width;
    if (idx >= total_size) return;

    int y = idx / params.target_width;
    int x = idx % params.target_width;

    const int height = params.input_height;
    const int width = params.input_width;
    const int target_height = params.target_height;
    const int target_width = params.target_width;
    const float* mean = params.mean;
    const float* std = params.std;

    int h_min = min(height, target_height);
    int w_min = min(width, target_width);
    int gap_h = target_height - h_min;

    int in_x = x;
    int in_y = y - gap_h;

    in_x = min(in_x, w_min - 1);
    in_y = max(min(in_y, h_min - 1), 0);

    int in_idx = (in_y * width + in_x) * 3;

    // Using local variables for mean and std
    float mean0 = mean[0], mean1 = mean[1], mean2 = mean[2];
    float std0 = std[0], std1 = std[1], std2 = std[2];

    // Normalize input data
    float normalized_pixel1_0 = ((input_data1[in_idx + 0] / 255.0f) - mean0) / std0;
    float normalized_pixel1_1 = ((input_data1[in_idx + 1] / 255.0f) - mean1) / std1;
    float normalized_pixel1_2 = ((input_data1[in_idx + 2] / 255.0f) - mean2) / std2;
    float normalized_pixel2_0 = ((input_data2[in_idx + 0] / 255.0f) - mean0) / std0;
    float normalized_pixel2_1 = ((input_data2[in_idx + 1] / 255.0f) - mean1) / std1;
    float normalized_pixel2_2 = ((input_data2[in_idx + 2] / 255.0f) - mean2) / std2;

    // Calculate output index
    int out_idx = y * target_width + x;

    // Write to output data
    output_data1[out_idx] = normalized_pixel1_0;
    output_data2[out_idx] = normalized_pixel2_0;
    output_data1[out_idx + total_size] = normalized_pixel1_1;
    output_data2[out_idx + total_size] = normalized_pixel2_1;
    output_data1[out_idx + 2 * total_size] = normalized_pixel1_2;
    output_data2[out_idx + 2 * total_size] = normalized_pixel2_2;
}

void FusedRightTopPadTransposeNormalizeImage(const float* d_input_data1, const float* d_input_data2, float* d_output_data1, float* d_output_data2, FusedParams params) {
    int target_height = params.target_height;
    int target_width = params.target_width;

    int threads_per_block = 256;
    int total_elements = target_height * target_width;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    FusedRightTopPadTransposeNormalizeKernel<<<blocks_per_grid, threads_per_block>>>(
      
        d_input_data1, d_input_data2, d_output_data1, d_output_data2, params
    );
}
