
# OpenStereo Deployment Guide

[English](README.md) | [中文](README.zh.md)

## Model Export

First, you need to convert the model checkpoint to a common IR (e.g., ONNX format). We provide the `export.py` script, which you can use to export a simplified ONNX model in the root directory of OpenStereo with the following command:

```bash
python deploy/export.py --config cfgs/psmnet/psmnet_kitti15.yaml --weights output/KittiDataset/PSMNet/psmnet_kitti15/default/ckpt/checkpoint_epoch_0.pth --device 0 --simplify --half --include onnx
```

If you want to deploy on NVIDIA devices using our C++ example code, you can add `engine` after the `--include` parameter to export TensorRT's IR:

```bash
python deploy/export.py --config cfgs/psmnet/psmnet_kitti15.yaml --weights output/KittiDataset/PSMNet/psmnet_kitti15/default/ckpt/checkpoint_epoch_0.pth --device 0 --simplify --half --include onnx engine
```

For more usage help, run:

```bash
python deploy/export.py -h
```

## Performance Evaluation

We provide the `trt_profile.sh` script to evaluate the model's performance on the device:

```bash
bash deploy/trt_profile.sh --onnx output/KittiDataset/PSMNet/psmnet_kitti15/default/ckpt/checkpoint_epoch_0.onnx --fp16 --verbose
bash deploy/trt_profile.sh --loadEngine output/KittiDataset/PSMNet/psmnet_kitti15/default/ckpt/checkpoint_epoch_0.engine --fp16 --verbose
```

## C++ Deployment Example

To run our C++ deployment example, you need to install some necessary third-party libraries:

```bash
apt-get update
apt-get install libyaml-cpp-dev libopencv-dev python3-opencv
```

Additionally, make sure you have installed the CUDA Toolkit and TensorRT:

- [CUDA Toolkit Download](https://developer.nvidia.com/cuda-downloads)
- [TensorRT Download](https://developer.nvidia.com/tensorrt)

Once ready, navigate to the `deploy/cpp` folder in the project root directory and run the following commands to build:

```bash
cd <OpenStereoROOT>/deploy/cpp
mkdir build && cd build
```

Build using the system library's TensorRT:

```bash
cmake .. && make
```

Or build using the downloaded TensorRT tar package:

```bash
cmake -DTENSORRT_ROOT=<path_to_tensorrt> .. && make
```

Then, you can perform inference with the following command:

```bash
./main <cfg_path> <engine_path> <left_image_path> <right_image_path> <options>
```

The required header files and dynamic libraries for deployment are packaged in `build/libopenstereo`, which can be conveniently included in your project.

Note: When building with the TensorRT tar package, ensure that the `LD_LIBRARY_PATH` is set correctly. You can check if the TensorRT dynamic library is linked correctly with the following command:

```bash
ldd main | grep libnv*
```
