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
#include <yaml-cpp/yaml.h>

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

    Transform(const std::map<std::string, TransformParams>& operations, bool verbose=false);
    std::unordered_map<std::string, cv::Mat> operator()(std::unordered_map<std::string, cv::Mat>& sample) const;

private:
    std::vector<TransformFunction> operations_;

    template<typename T>
    T GetParam(const TransformParams& params, const std::string& key) const;
};

inline std::map<std::string, Transform::TransformParams> parseConfig(const std::string& configPath, bool verbose=false) {
    YAML::Node config = YAML::LoadFile(configPath);
    std::map<std::string, Transform::TransformParams> operations;

    int knum_ops = 0;
    if (config["DATA_CONFIG"]["DATA_TRANSFORM"]["EVALUATING"]) {
        for (const auto& node : config["DATA_CONFIG"]["DATA_TRANSFORM"]["EVALUATING"]) {
            std::string op_name = node["NAME"].as<std::string>();

            // if operation name not in supported operation
            if (op_name == "ToTensor") continue;

            Transform::TransformParams params;

            for (const auto& param : node) {
                std::string param_name = param.first.as<std::string>();
                if (param_name == "NAME") continue; // Skip the NAME field
                if (param.second.IsSequence()) {
                    if (param_name == "SIZE") {
                        std::vector<int> vec = param.second.as<std::vector<int>>();
                        params[param_name] = vec;
                    } else {
                        std::vector<float> vec = param.second.as<std::vector<float>>();
                        params[param_name] = vec;
                    }
                } else if (param.second.IsScalar()) {
                    float value = param.second.as<float>();
                    params[param_name] = value;
                }
            }

            operations[op_name] = params;
            knum_ops++;
        }
    }

    if (!knum_ops || verbose) {
        std::cout << "Warning: no operation for preprocess" << std::endl;
    }

    return operations;
}

#endif // TRANSFORMS_H
