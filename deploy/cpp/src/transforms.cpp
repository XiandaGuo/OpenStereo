#include "transforms.h"

// =========================== OpenCV CPU Transformation ========================== //

RightTopPad::RightTopPad(int target_height, int target_width)
    : target_height_(target_height), target_width_(target_width) {}

std::unordered_map<std::string, cv::Mat> RightTopPad::operator()(std::unordered_map<std::string, cv::Mat>& sample) const {
    for (auto& item : sample) {
        const std::string& key = item.first;
        cv::Mat& image = item.second;

        if (key == "left_img" || key == "right_img") {
            image = PadImage(image, target_height_, target_width_, cv::BORDER_REPLICATE);
        } else if (key == "disp" || key == "disp_right" || key == "occ_mask" || key == "occ_mask_right") {
            image = PadImage(image, target_height_, target_width_, cv::BORDER_CONSTANT, cv::Scalar(0));
        }
    }
    return sample;
}

cv::Mat RightTopPad::PadImage(const cv::Mat& image, int target_height, int target_width, int borderType, const cv::Scalar& value) const {
    int h = std::min(image.rows, target_height);
    int w = std::min(image.cols, target_width);

    int pad_top = target_height - h;
    int pad_bottom = 0;
    int pad_left = 0;
    int pad_right = target_width - w;

    cv::Mat padded_image;
    cv::copyMakeBorder(image, padded_image, pad_top, pad_bottom, pad_left, pad_right, borderType, value);
    return padded_image;
}

TransposeImage::TransposeImage() {}

std::unordered_map<std::string, cv::Mat> TransposeImage::operator()(std::unordered_map<std::string, cv::Mat>& sample) const {
    for (auto& item : sample) {
        if (item.first == "left_img" || item.first == "right_img") {
            cv::Mat& image = item.second;
            image = Transpose(image);
        }
    }
    return sample;
}

/*
    tensor.transpose((2, 0, 1))
*/
cv::Mat TransposeImage::Transpose(const cv::Mat& image) const {
    // check empty
    if (image.empty()) {
        std::cerr << "Error: Input image is empty!" << std::endl;
        return cv::Mat();
    }

    // get shape
    int height = image.rows;
    int width = image.cols;
    int channels = image.channels();

    // creat chw_img
    cv::Mat chw_img(channels, height * width, CV_32FC1);

    // full the matrix
    // for (int c = 0; c < channels; ++c) {
    //     for (int h = 0; h < height; ++h) {
    //         for (int w = 0; w < width; ++w) {
    //             chw_img.at<float>(c, h * width + w) = image.at<cv::Vec3f>(h, w)[c];
    //         }
    //     }
    // }

    // full the matrix using pointers
    float* chw_data = chw_img.ptr<float>();
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            const cv::Vec3f& pixel = image.at<cv::Vec3f>(h, w);
            chw_data[h * width + w] = pixel[0];
            chw_data[height * width + h * width + w] = pixel[1];
            chw_data[2 * height * width + h * width + w] = pixel[2];
        }
    }
    
    return chw_img;
}

NormalizeImage::NormalizeImage(const std::vector<float>& mean, const std::vector<float>& std)
    : mean_(cv::Scalar(mean[0], mean[1], mean[2])), std_(cv::Scalar(std[0], std[1], std[2])) {}

std::unordered_map<std::string, cv::Mat> NormalizeImage::operator()(std::unordered_map<std::string, cv::Mat>& sample) const {
    for (auto& item : sample) {
        if (item.first == "left_img" || item.first == "right_img") {
            cv::Mat& image = item.second;
            Normalize(image);
        }
    }
    return sample;
}

void NormalizeImage::Normalize(cv::Mat& image) const {
    image = ((image / 255.0) - mean_) / std_;

    // image.convertTo(image, CV_32F, 1.0 / 255.0);

    // std::vector<cv::Mat> channels(3);
    // cv::split(image, channels);
    // for (int i = 0; i < 3; ++i) {
    //     channels[i] = (channels[i] - mean_[i]) / std_[i];
    // }
    // cv::merge(channels, image);
}

RightBottomCrop::RightBottomCrop(int target_height, int target_width)
    : target_height_(target_height), target_width_(target_width) {}

std::unordered_map<std::string, cv::Mat> RightBottomCrop::operator()(std::unordered_map<std::string, cv::Mat>& sample) const {
    for (auto& item : sample) {
        const std::string& key = item.first;
        cv::Mat& image = item.second;

        int h = image.rows;
        int w = image.cols;
        int crop_h = std::min(h, target_height_);
        int crop_w = std::min(w, target_width_);

        image = CropImage(image, crop_h, crop_w);
    }
    return sample;
}

cv::Mat RightBottomCrop::CropImage(const cv::Mat& image, int crop_height, int crop_width) const {
    int h = image.rows;
    int w = image.cols;

    cv::Rect crop_region(w - crop_width, h - crop_height, crop_width, crop_height);
    return image(crop_region);
}

CropOrPad::CropOrPad(int target_height, int target_width)
    : target_height_(target_height), target_width_(target_width), crop_fn_(target_height, target_width), pad_fn_(target_height, target_width) {}

std::unordered_map<std::string, cv::Mat> CropOrPad::operator()(std::unordered_map<std::string, cv::Mat>& sample) const {
    int h = sample.at("left_img").rows;
    int w = sample.at("left_img").cols;

    if (target_height_ > h || target_width_ > w) {
        sample = pad_fn_(sample);
    } else {
        sample = crop_fn_(sample);
    }

    return sample;
}

DivisiblePad::DivisiblePad(int by, const std::string& mode)
    : by_(by), mode_(mode) {}

std::unordered_map<std::string, cv::Mat> DivisiblePad::operator()(std::unordered_map<std::string, cv::Mat>& sample) const {
    int h = sample.at("left_img").rows;
    int w = sample.at("left_img").cols;

    int pad_h = (h % by_ != 0) ? (by_ - h % by_) : 0;
    int pad_w = (w % by_ != 0) ? (by_ - w % by_) : 0;

    int pad_top, pad_right, pad_bottom, pad_left;
    if (mode_ == "round") {
        pad_top = pad_h / 2;
        pad_bottom = pad_h - pad_top;
        pad_left = pad_w / 2;
        pad_right = pad_w - pad_left;
    } else if (mode_ == "tr") {
        pad_top = pad_h;
        pad_right = pad_w;
        pad_bottom = 0;
        pad_left = 0;
    } else {
        throw std::invalid_argument("No such mode for DivisiblePad");
    }

    for (auto& item : sample) {
        if (item.first == "left_img" || item.first == "right_img") {
            cv::copyMakeBorder(item.second, item.second, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_REPLICATE);
        } else if (item.first == "disp" || item.first == "disp_right" || item.first == "occ_mask" || item.first == "occ_mask_right") {
            cv::copyMakeBorder(item.second, item.second, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(0));
        }
    }

    std::vector<int> padding = {pad_top, pad_right, pad_bottom, pad_left};
    cv::Mat padding_mat(padding, true);
    sample["pad"] = padding_mat;

    return sample;
}

// =========================== Fused GPU Transformation =========================== //

FusedRightTopPadTransposeNormalize::FusedRightTopPadTransposeNormalize(const int target_height, const int target_width,
                                                                       const std::vector<float>& mean, const std::vector<float>& std) {
    params_.target_height = target_height;
    params_.target_width = target_width;
    params_.mean[0] = mean[0];
    params_.mean[1] = mean[1];
    params_.mean[2] = mean[2];
    params_.std[0] = std[0];
    params_.std[1] = std[1];
    params_.std[2] = std[2];

    output_bytes = params_.channels * target_height * target_width * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_input_data1, output_bytes));
    CUDA_CHECK(cudaMalloc(&d_input_data2, output_bytes));
    CUDA_CHECK(cudaMalloc(&d_output_data1, output_bytes));
    CUDA_CHECK(cudaMalloc(&d_output_data2, output_bytes));
}

inline std::unordered_map<std::string, float*> FusedRightTopPadTransposeNormalize::operator()(std::unordered_map<std::string, cv::Mat>& sample) {
    auto input_image1 = sample["left_img"];
    auto input_image2 = sample["right_img"];

    input_bytes = params_.channels * input_image1.rows * input_image1.cols * sizeof(float);

    params_.input_height = input_image1.rows;
    params_.input_width = input_image1.cols;

    cudaMemcpy(d_input_data1, input_image1.ptr<float>(), input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_data2, input_image2.ptr<float>(), input_bytes, cudaMemcpyHostToDevice);

    FusedRightTopPadTransposeNormalizeImage(d_input_data1, d_input_data2, d_output_data1, d_output_data2, params_);

    std::unordered_map<std::string, float*> preprocess_result = {
        {"left_img", d_output_data1},
        {"right_img", d_output_data2}
    };

    return preprocess_result;
}

// ================================== Transform =================================== //

Transform::Transform(const std::vector<std::pair<std::string, Transform::TransformParams>>& operations, bool verbose) {
    if (verbose) {std::cout << "================= Transform Info =================" << std::endl;}

    std::vector<std::string> operator_sequence;
    std::vector<TransformFunction> temp_operations;

    for (const auto& op : operations) {
        const std::string& op_name = op.first;
        const auto& params = op.second;
        if (verbose) {std::cout << "NAME: " << op_name;}
        operator_sequence.push_back(op_name);

        if (op_name == "RightTopPad") {
            std::vector<int> size = GetParam<std::vector<int>>(params, "SIZE");
            int target_height = size[0], target_width = size[1];
            temp_operations.push_back(RightTopPad(target_height, target_width));
            if (verbose) {
                std::cout << ", SIZE: [" << target_height << ", " << target_width << "]" << std::endl;
            }
        } else if (op_name == "TransposeImage") {
            temp_operations.push_back(TransposeImage());
            if (verbose) {
                std::cout << std::endl;
            }
        } else if (op_name == "NormalizeImage") {
            std::vector<float> mean = GetParam<std::vector<float>>(params, "MEAN");
            std::vector<float> std = GetParam<std::vector<float>>(params, "STD");
            temp_operations.push_back(NormalizeImage(mean, std));
            if (verbose) {
                std::cout << ", MEAN: [" << mean[0] << ", " << mean[1] << ", " << mean[2] << "]"
                          << ", STD: [" << std[0] << ", " << std[1] << ", " << std[2] << "]" << std::endl;
            }
        } else if (op_name == "RightBottomCrop") {
            std::vector<int> size = GetParam<std::vector<int>>(params, "SIZE");
            int target_height = size[0], target_width = size[1];
            temp_operations.push_back(RightBottomCrop(target_height, target_width));
            if (verbose) {
                std::cout << ", SIZE: [" << target_height << ", " << target_width << "]" << std::endl;
            }
        } else if (op_name == "DivisiblePad") {
            int by = GetParam<int>(params, "BY");
            std::string mode = "round";
            try {
                mode = GetParam<std::string>(params, "MODE");
                temp_operations.push_back(DivisiblePad(by, mode));
            } catch(const std::exception& e) {
                std::cerr << e.what() << " used default mode" << '\n';
                temp_operations.push_back(DivisiblePad(by, mode));
            }
            if (verbose) {
                std::cout << ", BY: " << by << ", MODE: " << mode << std::endl;
            }
        } else if (op_name == "CropOrPad") {
            std::vector<int> size = GetParam<std::vector<int>>(params, "SIZE");
            int target_height = size[0], target_width = size[1];
            temp_operations.push_back(CropOrPad(target_height, target_width));
            if (verbose) {
                std::cout << ", SIZE: [" << target_height << ", " << target_width << "]" << std::endl;
            }
        } else if (op_name == "FusedRightTopPadTransposeNormalize") {
            std::vector<int> size = GetParam<std::vector<int>>(params, "SIZE");
            std::vector<float> mean = GetParam<std::vector<float>>(params, "MEAN");
            std::vector<float> std = GetParam<std::vector<float>>(params, "STD");
            int target_height = size[0], target_width = size[1];
            temp_operations.push_back(FusedRightTopPadTransposeNormalize(target_height, target_width, mean, std));
            if (verbose) {
                std::cout << ", SIZE: [" << target_height << ", " << target_width << "]"
                          << ", MEAN: [" << mean[0] << ", " << mean[1] << ", " << mean[2] << "]"
                          << ", STD: [" << std[0] << ", " << std[1] << ", " << std[2] << "]" << std::endl;
            }
        } else {
            throw std::invalid_argument("Unsupported operation: " + op_name);
        }
    }

    // fused op
    auto it = supported_fusions.find(operator_sequence);
    if (it != supported_fusions.end()) {
        if (it->second == "FusedRightTopPadTransposeNormalize") {
            std::vector<int> size = GetParam<std::vector<int>>(operations[0].second, "SIZE");
            std::vector<float> mean = GetParam<std::vector<float>>(operations[2].second, "MEAN");
            std::vector<float> std = GetParam<std::vector<float>>(operations[2].second, "STD");
            int target_height = size[0], target_width = size[1];

            operations_.clear();
            operations_.push_back(FusedRightTopPadTransposeNormalize(target_height, target_width, mean, std));
        }
    } else {
        operations_ = std::move(temp_operations);
    }
}

PreprocessType Transform::operator()(std::unordered_map<std::string, cv::Mat>& sample) const {
    PreprocessType result;
    for (const auto& operation : operations_) {
        result = operation(sample);
    }
    return result;
}

template<typename T>
T Transform::GetParam(const TransformParams& params, const std::string& key) const {
    auto it = params.find(key);
    if (it != params.end()) {
        try {
            return std::get<T>(it->second);
        } catch (const std::bad_variant_access&) {
            throw std::invalid_argument("Parameter type mismatch for key: " + key);
        }
    } else {
        throw std::invalid_argument("Missing required parameter: " + key);
    }
}
