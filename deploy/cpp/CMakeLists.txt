cmake_minimum_required(VERSION 3.10)
project(main)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

# Find yaml-cpp
find_package(yaml-cpp REQUIRED)
include_directories(${yaml-cpp_INCLUDE_DIRS})

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS} include)
message(STATUS "CUDAToolkit_LIBRARY_ROOT: ${CUDAToolkit_LIBRARY_ROOT}")

# Include TensorRT
# set(TENSORRT_ROOT "" CACHE PATH "Path to TensorRT installation")
# if(NOT TENSORRT_ROOT)
#     message(FATAL_ERROR "Please set TENSORRT_ROOT to the root directory of your TensorRT installation")
# endif()
# include_directories(${TENSORRT_ROOT}/include)
# Find TensorRT

set(TENSORRT_ROOT "" CACHE PATH "Path to TensorRT installation")

# Check if TENSORRT_ROOT is set
if(NOT TENSORRT_ROOT)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        PATHS /usr/local/include
              /usr/include
              /usr/local/TensorRT/include
              /usr/local/tensorrt/include
              /opt/TensorRT/include
              /opt/tensorrt/include
              $ENV{HOME}/TensorRT/include
              $ENV{HOME}/tensorrt/include
    )

    find_library(TENSORRT_LIBRARY nvinfer
        PATHS /usr/local/lib
              /usr/lib
              /usr/local/TensorRT/lib
              /usr/local/tensorrt/lib
              /opt/TensorRT/lib
              /opt/tensorrt/lib
              $ENV{HOME}/TensorRT/lib
              $ENV{HOME}/tensorrt/lib
    )

    find_library(TENSORRT_ONNXPARSER nvonnxparser
        PATHS /usr/local/lib
              /usr/lib
              /usr/local/TensorRT/lib
              /usr/local/tensorrt/lib
              /opt/TensorRT/lib
              /opt/tensorrt/lib
              $ENV{HOME}/TensorRT/lib
              $ENV{HOME}/tensorrt/lib
    )
    
    if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY AND TENSORRT_ONNXPARSER)
        message(STATUS "Found TensorRT includes in: ${TENSORRT_INCLUDE_DIR}")
        get_filename_component(TENSORRT_LIB_DIR ${TENSORRT_LIBRARY} DIRECTORY)
        message(STATUS "Found TensorRT libraries in: ${TENSORRT_LIB_DIR}")
        set(TENSORRT_ROOT ${TENSORRT_LIB_DIR} CACHE PATH "Path to TensorRT installation" FORCE)
    else()
        message(FATAL_ERROR "TensorRT not found. Please set TENSORRT_ROOT to the root directory of your TensorRT installation.")
    endif()

else()
    if(EXISTS "${TENSORRT_ROOT}/include/NvInfer.h" AND EXISTS "${TENSORRT_ROOT}/lib/libnvinfer.so")
        message(STATUS "Using specified TensorRT root: ${TENSORRT_ROOT}")
        set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include")
        set(TENSORRT_LIB_DIR "${TENSORRT_ROOT}/lib")
    else()
        message(FATAL_ERROR "TensorRT not found. Please set TENSORRT_ROOT to the root directory of your TensorRT installation.")
    endif()
endif()

include_directories(${TENSORRT_INCLUDE_DIR})

# Show Vars
# get_cmake_property(_variableNames VARIABLES)
# foreach (_variableName ${_variableNames})
#     message(STATUS "${_variableName}=${${_variableName}}")
# endforeach()

# Add src
file(GLOB SOURCES "src/*.cpp" "*.cpp")

# Add exec
add_executable(main ${SOURCES})

# Add lib
add_library(openstereo SHARED ${SOURCES})

# Link libraries
target_link_libraries(openstereo ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${YAML_CPP_LIBRARIES} cuda ${CUDA_CUDART} ${TENSORRT_LIB_DIR}/libnvinfer.so ${TENSORRT_LIB_DIR}/libnvonnxparser.so)
target_link_libraries(main openstereo)

# Set default install directory to current build directory
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/libopenstereo" CACHE PATH "Installation Directory" FORCE)
endif()

# install rules
install(TARGETS main DESTINATION bin)
install(TARGETS openstereo DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

# Add a POST_BUILD event to automatically run install after main is built
add_custom_command(TARGET main POST_BUILD
    COMMAND ${CMAKE_COMMAND} --build . --target install
    COMMENT "Running install after build..."
)