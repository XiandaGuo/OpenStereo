#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <string>
#include <fstream>
#include <cassert>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
#endif  // CUDA_CHECK

#define CHECK_RETURN_W_MSG(status, val, errMsg)                                                                                \
    do                                                                                                                         \
    {                                                                                                                          \
        if (!(status))                                                                                                         \
        {                                                                                                                      \
            sample::gLogError << errMsg << " Error in " << __FILE__ << ", function " << FN_NAME << "(), line " << __LINE__     \
                      << std::endl;                                                                                            \
            return val;                                                                                                        \
        }                                                                                                                      \
    } while (0)

#define CHECK_PATH(path) \
    (std::ifstream(path).good() ? true : (std::cerr << "Error: Path does not exist: " << path << std::endl, false))

#endif  // UTILS_H_
