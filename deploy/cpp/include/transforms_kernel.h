#ifndef TRANSFORMS_KERNEL_H
#define TRANSFORMS_KERNEL_H

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

// =========================== OpenCV CPU Transformation ========================== //

void PadImageCUDA(cv::Mat& image, int target_height, int target_width, float fill_value[3], int border_type);

// =========================== Fused GPU Transformation =========================== //
// =================== FusedRightTopPadTransposeNormalizeKernel =================== //

struct FusedParams {
    int channels = 3;
    int target_height;
    int target_width;
    float mean[3];
    float std[3];

    int input_height;
    int input_width;
};

void FusedRightTopPadTransposeNormalizeImage(const float* d_input_data1, const float* d_input_data2, float* d_output_data1, float* d_output_data2, FusedParams params);

#endif // TRANSFORMS_KERNEL_H