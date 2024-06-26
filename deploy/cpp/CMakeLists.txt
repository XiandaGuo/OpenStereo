cmake_minimum_required(VERSION 3.10)
project(main)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS} include)
message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")

# Include TensorRT
include_directories(/ryn/TensorRT-Tars/TensorRT-10.0.1.6/include/)

# Add src
file(GLOB SOURCES "src/*.cpp" "*.cpp")

# Add exec
add_executable(main ${SOURCES})

# Add lib
add_library(openstereo SHARED ${SOURCES})

# Link libraries
target_link_libraries(openstereo ${OpenCV_LIBS} ${CUDA_LIBRARIES} cuda /usr/local/cuda/lib64/libcudart.so /ryn/TensorRT-Tars/TensorRT-10.0.1.6/lib/libnvinfer.so /ryn/TensorRT-Tars/TensorRT-10.0.1.6/lib/libnvonnxparser.so)
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