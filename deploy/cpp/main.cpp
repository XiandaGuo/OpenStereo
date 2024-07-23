#include <chrono>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "transforms.h"
#include "inference.h"
#include "utils.h"

// Global variables for options
bool VERBOSE = 0;
bool SHOW = 0;
bool DEBUG = 0;
int WARMUP_TIME = 10;

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << "<cfg_path> <engine_path> <left_image_path> <right_image_path> <options>" << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  --verbose, -v      Enable verbose output" << std::endl;
        std::cerr << "  --warmup <value>   Set warmup time to <value> (integer)" << std::endl;
        std::cerr << "  --show             Show images" << std::endl;
        std::cerr << "  --dbg              Enable debugging mode" << std::endl;
        return 1;
    }

    std::string cfg_path = argv[1];
    std::string engine_path = argv[2];
    std::string left_image_path = argv[3];
    std::string right_image_path = argv[4];

    // Deal with options
    for (int i = 5; i < argc; ++i) {
        std::string option = argv[i];
        if (option == "--verbose" || option == "-v") {
            VERBOSE = 1;
        } else if (option == "--warmup") {
            if (i + 1 < argc) {
                std::string warmup_value = argv[++i];
                std::istringstream ss(warmup_value);
                if (!(ss >> WARMUP_TIME)) {
                    std::cerr << "Error: --warmup requires an integer value, got " << warmup_value << std::endl;
                    return 1;
                }
                std::cout << "Warmup count set to: " << WARMUP_TIME << " , default set to 10" << std::endl;
            } else {
                std::cerr << "Error: --warmup requires a value" << std::endl;
                return 1;
            }
        } else if (option == "--show") {
            SHOW = 1;
        } else if (option == "--dbg") {
            DEBUG = 1;
        } else {
            std::cerr << "Unknown option: " << option << std::endl;
            return 1;
        }
    }

    // Check if the image paths and engine path are valid
    if (!CHECK_PATH(cfg_path) || !CHECK_PATH(engine_path) || !CHECK_PATH(left_image_path) || !CHECK_PATH(right_image_path)) {
        return -1;
    }

    // Load TensorRT engine
    InferenceEngine engine(engine_path);

    // Load images and transform
    cv::Mat left_img_bgr = cv::imread(left_image_path, cv::IMREAD_COLOR);
    cv::Mat right_img_bgr = cv::imread(right_image_path, cv::IMREAD_COLOR);

    if (left_img_bgr.empty() || right_img_bgr.empty()) {
        std::cerr << "Could not open or find the images!" << std::endl;
        return 1;
    }

    // debug code
    if (DEBUG) {
        cv::Rect cropRegion(0, 0, 512, 256);
        left_img_bgr = left_img_bgr(cropRegion);
        right_img_bgr = right_img_bgr(cropRegion);
        cv::imwrite("left_img.png", left_img_bgr);
        cv::imwrite("right_img.png", right_img_bgr);
    }

    cv::Mat left_img_rgb;
    cv::Mat right_img_rgb;
    cv::cvtColor(left_img_bgr, left_img_rgb, cv::COLOR_BGR2RGB);
    cv::cvtColor(right_img_bgr, right_img_rgb, cv::COLOR_BGR2RGB);

    left_img_rgb.convertTo(left_img_rgb, CV_32FC3);
    right_img_rgb.convertTo(right_img_rgb, CV_32FC3);

    std::unordered_map<std::string, cv::Mat> sample = {
        {"left_img", left_img_rgb},
        {"right_img", right_img_rgb}
    };

    // Extra transform sequence from cfg file
    auto operations = parseConfig(cfg_path, VERBOSE);
    // *** You can also create an operations sequence manually as shown in the code below ***
    // std::map<std::string, Transform::TransformParams> operations;
    // operations["RightTopPad"] = {{"SIZE", std::vector<int>{384, 1248}}};
    // operations["TransposeImage"] = Transform::TransformParams();
    // operations["NormalizeImage"] = {{"MEAN", std::vector<float>{0.485f, 0.456f, 0.406f}},
    //                                 {"STD", std::vector<float>{0.229f, 0.224f, 0.225f}}};
    Transform transform(operations, VERBOSE);

    auto start_preprocess = std::chrono::high_resolution_clock::now();

    sample = transform(sample);

    auto end_preprocess = std::chrono::high_resolution_clock::now();
    auto duration_preprocess = std::chrono::duration_cast<std::chrono::milliseconds>(end_preprocess - start_preprocess);

    // Try run and warm up engine
    try {
        for (int i = 0; i < WARMUP_TIME; i++) {
            engine.run(sample);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error running inference: " << e.what() << std::endl;
        return 1;
    }

    // Run engine
    auto start_inference = std::chrono::high_resolution_clock::now();

    std::unordered_map<std::string, cv::Mat> output = engine.run(sample);

    auto end_inference = std::chrono::high_resolution_clock::now();
    auto duration_inference = std::chrono::duration_cast<std::chrono::milliseconds>(end_inference - start_inference);

    std::cout << "==================== TimeInfo ====================" << std::endl
              << "preprocess: " << duration_preprocess.count() << " ms" << std::endl
              << "inference : " << duration_inference.count() << " ms" << std::endl;

    if (SHOW) {
        // Show inference result
        cv::imshow("Disparity Prediction", output["color_normalized_disp_pred"]);
        cv::waitKey(0);
    }

    // Save inference result
    std::cout << "==================== MiscInfo ====================" << std::endl;
    std::string output_path = "disp_pred.png";
    if (cv::imwrite(output_path, output["color_normalized_disp_pred"])) {
        std::cout << "Image saved successfully: " << output_path << std::endl;
    } else {
        std::cerr << "Failed to save image: " << output_path << std::endl;
    }

    return 0;
}
