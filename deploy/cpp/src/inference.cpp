#include "inference.h"
#include "utils.h"
#include <fstream>
#include <iostream>

#define DEVICE_ID 0
// #define VERBOSE

size_t volume(const nvinfer1::Dims& dims) {  
    size_t vol = 1;  
    for (int i = 0; i < dims.nbDims; ++i) {  
        vol *= dims.d[i];  
    }  
    return vol;  
}

size_t getElementSize(nvinfer1::DataType type) {  
    switch (type) {  
        case nvinfer1::DataType::kFLOAT: return sizeof(float);  
        case nvinfer1::DataType::kHALF: return sizeof(half);
        default: throw std::runtime_error("Unsupported data type");  
    }  
}

Logger InferenceEngine::gLogger;

InferenceEngine::InferenceEngine(const std::string& engine_file) {
    cudaError_t status = cudaSetDevice(DEVICE_ID);
    if (status != cudaSuccess) {
        std::cerr << "Error setting CUDA device: " << cudaGetErrorString(status) << std::endl;
        return;
    }

    loadEngine(engine_file);

    runtime_ = nvinfer1::createInferRuntime(gLogger);
    if (!runtime_) throw std::runtime_error("Failed to create TensorRT runtime");

    engine_ = runtime_->deserializeCudaEngine(engine_data_.data(), engine_data_.size());
    if (!engine_) throw std::runtime_error("Failed to create TensorRT engine");

    context_ = engine_->createExecutionContext();
    if (!context_) throw std::runtime_error("Failed to create TensorRT context");

    // Allocate buffers
    allocateBuffers();

    // Ensure the tensor addresses are set correctly
    context_->setTensorAddress("left_img", buffers_[0]);
    context_->setTensorAddress("right_img", buffers_[1]);
    context_->setTensorAddress("disp_pred", buffers_[2]);

    #ifdef VERBOSE
        // Show information about tensorrt engine
        showEngineInfo();
    #endif

    CUDA_CHECK(cudaStreamCreate(&stream_));
}

InferenceEngine::~InferenceEngine() {
    cudaStreamDestroy(stream_);

    // Destroy the engine
    delete context_;
    delete engine_;
    delete runtime_;
}

void InferenceEngine::loadEngine(const std::string& engine_file) {
    std::ifstream file(engine_file, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open engine file");

    file.seekg(0, std::ios::end);
    const size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    engine_data_.resize(size);
    file.read(engine_data_.data(), size);
    file.close();
}

void InferenceEngine::showEngineInfo() {
    std::cout << "=================== EngineInfo ===================" << std::endl;

   // Get number of io tensors (inputs + outputs)
    int nbIOTensors = engine_->getNbIOTensors();

    for (int i = 0; i < nbIOTensors; ++i) {
        std::cout << "IOTensor " << i << ":" << std::endl;

        // Get io tensor name
        std::cout << "  Name: " << engine_->getIOTensorName(i) << std::endl;

        // Get io tensor dimensions
        auto dims = engine_->getTensorShape(engine_->getIOTensorName(i));
        std::cout << "  Dimensions: ";
        for (int i = 0; i < dims.nbDims; ++i) {
            std::cout << dims.d[i] << " ";
        }
        std::cout << std::endl;

        // Get io tensor data type
        auto dtype = engine_->getTensorDataType(engine_->getIOTensorName(i));
        std::string dtypeStr;
        switch (dtype) {
            case nvinfer1::DataType::kFLOAT: dtypeStr = "FLOAT"; break;
            case nvinfer1::DataType::kHALF: dtypeStr = "HALF"; break;
            case nvinfer1::DataType::kINT8: dtypeStr = "INT8"; break;
            case nvinfer1::DataType::kINT32: dtypeStr = "INT32"; break;
            case nvinfer1::DataType::kBOOL: dtypeStr = "BOOL"; break;
            default: dtypeStr = "UNKNOWN"; break;
        }
        std::cout << "  DataType: " << dtypeStr << std::endl;

        // Check if io tensor is input or output
        if (engine_->getTensorIOMode(engine_->getIOTensorName(i)) == nvinfer1::TensorIOMode::kINPUT) {
            std::cout << "  TensorIOMode: Input" << std::endl;
        } else {
            std::cout << "  TensorIOMode: Output" << std::endl;
        }
    }
    std::cout << "**Note: DataType is the type of IOTensor, not the type of runtime." << std::endl;
    std::cout << "==================================================" << std::endl;
}

void InferenceEngine::allocateBuffers() {
    buffers_.resize(engine_->getNbIOTensors());

    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        auto dims = engine_->getTensorShape(engine_->getIOTensorName(i));
        auto type = engine_->getTensorDataType(engine_->getIOTensorName(i));
        size_t size = volume(dims) * getElementSize(type);

        cudaMalloc(&buffers_[i], size);
    }
}

void InferenceEngine::preprocess(const std::unordered_map<std::string, cv::Mat>& sample) {
    // host2device
    for (const auto& item : sample) {
        const std::string& key = item.first;
        const cv::Mat& image = item.second;

        if (key == "left_img") {
            cudaMemcpyAsync(buffers_[0], image.data, image.total() * image.elemSize(), cudaMemcpyHostToDevice, stream_);
        } else if (key == "right_img") {
            cudaMemcpyAsync(buffers_[1], image.data, image.total() * image.elemSize(), cudaMemcpyHostToDevice, stream_);
        }
    }
}

std::unordered_map<std::string, cv::Mat> InferenceEngine::postprocess() {
    // get inference result
    std::unordered_map<std::string, cv::Mat> output;
    auto disp_pred_dims = engine_->getTensorShape(engine_->getIOTensorName(engine_->getNbIOTensors() - 1));
    cv::Mat disp_pred(disp_pred_dims.d[1], disp_pred_dims.d[2], CV_32FC1); // Adjust the size and type accordingly

    // Memcpy from device output buffer to host output buffer
    cudaMemcpyAsync(disp_pred.data, buffers_[2], disp_pred.total() * disp_pred.elemSize(), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    output["disp_pred"] = disp_pred;

    if (output["disp_pred"].empty()) {
        std::cerr << "Error: disp_pred cv::Mat is empty!" << std::endl;
        return output;
    }

    double minVal, maxVal;
    cv::minMaxLoc(output["disp_pred"], &minVal, &maxVal);
    cv::Mat normalized_disp_pred;
    output["disp_pred"].convertTo(normalized_disp_pred, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
    output["normalized_disp_pred"] = normalized_disp_pred;

    cv::Mat color_normalized_disp_pred;
    cv::applyColorMap(normalized_disp_pred, color_normalized_disp_pred, cv::COLORMAP_JET);
    output["color_normalized_disp_pred"] = color_normalized_disp_pred;

    return output;
}

std::unordered_map<std::string, cv::Mat> InferenceEngine::run(const std::unordered_map<std::string, cv::Mat>& sample) {
    preprocess(sample);

    // Enqueue the inference
    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("Failed to enqueue inference");
    }

    return postprocess();
}
