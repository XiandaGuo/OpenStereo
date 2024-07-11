#ifndef TRANSFORMS_H
#define TRANSFORMS_H

#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <variant>

class RightTopPad {
public:
    RightTopPad(int target_height, int target_width);
    std::unordered_map<std::string, cv::Mat> operator()(std::unordered_map<std::string, cv::Mat>& sample) const;

private:
    int target_height_;
    int target_width_;
    cv::Mat PadImage(const cv::Mat& image, int target_height, int target_width, int borderType, const cv::Scalar& value = cv::Scalar()) const;
};

class TransposeImage {
public:
    TransposeImage();
    std::unordered_map<std::string, cv::Mat> operator()(std::unordered_map<std::string, cv::Mat>& sample) const;

private:
    cv::Mat Transpose(const cv::Mat& image) const;
};

class NormalizeImage {
public:
    NormalizeImage(const std::vector<float>& mean, const std::vector<float>& std);
    std::unordered_map<std::string, cv::Mat> operator()(std::unordered_map<std::string, cv::Mat>& sample) const;

private:
    cv::Scalar mean_;
    cv::Scalar std_;
    void Normalize(cv::Mat& image) const;
};

class RightBottomCrop {
public:
    RightBottomCrop(int target_height, int target_width);
    std::unordered_map<std::string, cv::Mat> operator()(std::unordered_map<std::string, cv::Mat>& sample) const;

private:
    int target_height_;
    int target_width_;
    cv::Mat CropImage(const cv::Mat& image, int crop_height, int crop_width) const;
};

class CropOrPad {
public:
    CropOrPad(int target_height, int target_width);
    std::unordered_map<std::string, cv::Mat> operator()(std::unordered_map<std::string, cv::Mat>& sample) const;

private:
    int target_height_;
    int target_width_;
    RightBottomCrop crop_fn_;
    RightTopPad pad_fn_;
};

class DivisiblePad {
public:
    DivisiblePad(int by, const std::string& mode);
    std::unordered_map<std::string, cv::Mat> operator()(std::unordered_map<std::string, cv::Mat>& sample) const;

private:
    int by_;
    std::string mode_;
};

class Transform {
public:
    using ParamValue = std::variant<int, float, std::string, std::vector<int>, std::vector<float>>;
    using TransformParams = std::unordered_map<std::string, ParamValue>;
    using TransformFunction = std::function<std::unordered_map<std::string, cv::Mat>(std::unordered_map<std::string, cv::Mat>&)>;

    Transform(const std::map<std::string, TransformParams>& operations);
    std::unordered_map<std::string, cv::Mat> operator()(std::unordered_map<std::string, cv::Mat>& sample) const;

private:
    std::vector<TransformFunction> operations_;

    template<typename T>
    T GetParam(const TransformParams& params, const std::string& key) const;
};

#endif // TRANSFORMS_H
