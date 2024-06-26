#ifndef TRANSFORMS_H
#define TRANSFORMS_H

#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <string>

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
    NormalizeImage(const cv::Scalar& mean, const cv::Scalar& std);
    std::unordered_map<std::string, cv::Mat> operator()(std::unordered_map<std::string, cv::Mat>& sample) const;

private:
    cv::Scalar mean_;
    cv::Scalar std_;
    void Normalize(cv::Mat& image) const;
};

#endif // TRANSFORMS_H
