#include "transforms.h"

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
        if (item.first == "left" || item.first == "right") {
            cv::Mat& image = item.second;
            Normalize(image);
        }
    }
    return sample;
}

void NormalizeImage::Normalize(cv::Mat& image) const {
    image = (image - mean_) / std_;
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
    int h = sample.at("left").rows;
    int w = sample.at("left").cols;

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
    int h = sample.at("left").rows;
    int w = sample.at("left").cols;

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
        if (item.first == "left" || item.first == "right") {
            cv::copyMakeBorder(item.second, item.second, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_REPLICATE);
        } else if (item.first == "disp" || item.first == "disp_right" || item.first == "occ_mask" || item.first == "occ_mask_right") {
            cv::copyMakeBorder(item.second, item.second, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(0));
        }
    }

    std::vector<int> padding = { pad_top, pad_right, pad_bottom, pad_left };
    cv::Mat padding_mat(padding, true);
    sample["pad"] = padding_mat;

    return sample;
}

Transform::Transform(const std::map<std::string, TransformParams>& operations) {
    for (const auto& op : operations) {
        const std::string& op_name = op.first;
        const auto& params = op.second;

        if (op_name == "RightTopPad") {
            int target_height = GetParam<int>(params, "target_height");
            int target_width = GetParam<int>(params, "target_width");
            operations_.emplace_back(RightTopPad(target_height, target_width));
        } else if (op_name == "RightBottomCrop") {
            int target_height = GetParam<int>(params, "target_height");
            int target_width = GetParam<int>(params, "target_width");
            operations_.emplace_back(RightBottomCrop(target_height, target_width));
        } else if (op_name == "TransposeImage") {
            operations_.emplace_back(TransposeImage());
        } else if (op_name == "NormalizeImage") {
            std::vector<float> mean = GetParam<std::vector<float>>(params, "mean");
            std::vector<float> std = GetParam<std::vector<float>>(params, "std");
            operations_.emplace_back(NormalizeImage(mean, std));
        } else if (op_name == "DivisiblePad") {
            int by = GetParam<int>(params, "by");
            std::string mode = GetParam<std::string>(params, "mode");
            operations_.emplace_back(DivisiblePad(by, mode));
        } else if (op_name == "CropOrPad") {
            int target_height = GetParam<int>(params, "target_height");
            int target_width = GetParam<int>(params, "target_width");
            operations_.emplace_back(CropOrPad(target_height, target_width));
        } else {
            throw std::invalid_argument("Unsupported operation: " + op_name);
        }
    }
}

std::unordered_map<std::string, cv::Mat> Transform::operator()(std::unordered_map<std::string, cv::Mat>& sample) const {
    for (const auto& operation : operations_) {
        sample = operation(sample);
    }
    return sample;
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
