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

#include "transforms_kernel.h"
#include "utils.h"

// =========================== OpenCV CPU Transformation ========================== //

class RightTopPad {
public:
    RightTopPad(int target_height, int target_width);
    std::unordered_map<std::string, cv::Mat> operator()(std::unordered_map<std::string, cv::Mat>& sample) const;

private:
    int BORDER_CONSTANT = 0;
    int BORDER_REPLICATE = 1;

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

// ==================== Fused GPU Transformation ==================== //

struct OperationSequenceHash {
    std::size_t operator()(const std::vector<std::string>& seq) const {
        std::size_t hash = 0;
        for (const auto& s : seq) {
            hash ^= std::hash<std::string>{}(s) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

const std::unordered_map<std::vector<std::string>, std::string, OperationSequenceHash> supported_fusions = {
    {{"RightTopPad", "TransposeImage", "NormalizeImage"}, "FusedRightTopPadTransposeNormalize"}
};

class FusedRightTopPadTransposeNormalize {
public:
    FusedRightTopPadTransposeNormalize(const int target_height, const int target_width,
                                       const std::vector<float>& mean, const std::vector<float>& std);
    std::unordered_map<std::string, float*> operator()(std::unordered_map<std::string, cv::Mat>& sample);

private:
    FusedParams params_;

    size_t input_bytes;
    size_t output_bytes;
    float* d_input_data1 = nullptr;
    float* d_input_data2 = nullptr;
    float* d_output_data1 = nullptr;
    float* d_output_data2 = nullptr;
};

// =========================== Transform ============================ //

using PreprocessType = std::variant<std::unordered_map<std::string, cv::Mat>, std::unordered_map<std::string, float*>>;

class Transform {
public:
    using ParamValue = std::variant<int, float, std::string, std::vector<int>, std::vector<float>>;
    using TransformParams = std::unordered_map<std::string, ParamValue>;
    using TransformFunction = std::function<PreprocessType(std::unordered_map<std::string, cv::Mat>&)>;

    Transform(const std::vector<std::pair<std::string, Transform::TransformParams>>& operations, bool verbose=false);
    PreprocessType operator()(std::unordered_map<std::string, cv::Mat>& sample) const;

private:
    std::vector<TransformFunction> operations_;

    template<typename T>
    T GetParam(const TransformParams& params, const std::string& key) const;
};

inline std::vector<std::pair<std::string, Transform::TransformParams>> parseConfig(const std::string& configPath, bool verbose=false) {
    YAML::Node config = YAML::LoadFile(configPath);
    std::vector<std::pair<std::string, Transform::TransformParams>> operations;

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

            operations.emplace_back(op_name, params);
            knum_ops++;
        }
    }

    if (!knum_ops && verbose) {
        std::cout << "Warning: no operation for preprocess" << std::endl;
    }

    return operations;
}

#endif // TRANSFORMS_H
