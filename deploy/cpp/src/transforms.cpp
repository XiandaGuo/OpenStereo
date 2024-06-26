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

NormalizeImage::NormalizeImage(const cv::Scalar& mean, const cv::Scalar& std)
    : mean_(mean), std_(std) {}

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
