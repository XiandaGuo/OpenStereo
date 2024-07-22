# OpenStereo 部署指南

[English](README.md) | [中文](README.zh.md)

## 模型导出

首先，你需要将模型的 checkpoint 转换为通用的 IR（如 ONNX 格式）。我们提供了 `export.py` 脚本，可以通过以下指令在 OpenStereo 的根目录下导出简化后的 ONNX 模型：

```bash
python deploy/export.py --config cfgs/psmnet/psmnet_kitti15.yaml --weights output/KittiDataset/PSMNet/psmnet_kitti15/default/ckpt/checkpoint_epoch_0.pth --device 0 --simplify --half --include onnx
```

如果希望使用我们的 C++ 示例代码在 NVIDIA 设备上进行部署，可以在 `--include` 参数后添加 `engine` 以导出 TensorRT 的 IR：

```bash
python deploy/export.py --config cfgs/psmnet/psmnet_kitti15.yaml --weights output/KittiDataset/PSMNet/psmnet_kitti15/default/ckpt/checkpoint_epoch_0.pth --device 0 --simplify --half --include onnx engine
```

更多使用帮助，请运行：

```bash
python deploy/export.py -h
```

## 性能评估

我们提供了 `trt_profile.sh` 脚本，用于评估模型在设备上的性能：

```bash
bash deploy/trt_profile.sh --onnx output/KittiDataset/PSMNet/psmnet_kitti15/default/ckpt/checkpoint_epoch_0.onnx --fp16 --verbose
bash deploy/trt_profile.sh --loadEngine output/KittiDataset/PSMNet/psmnet_kitti15/default/ckpt/checkpoint_epoch_0.engine --fp16 --verbose
```

## C++ 部署示例

要运行我们的 C++ 部署示例，需要安装一些必要的第三方库：

```bash
apt-get update
apt-get install libyaml-cpp-dev libopencv-dev python3-opencv
```

此外，还需确保已安装 CUDA Toolkit 和 TensorRT：

- [CUDA Toolkit 下载](https://developer.nvidia.com/cuda-downloads)
- [TensorRT 下载](https://developer.nvidia.com/tensorrt)

准备就绪后，进入项目根目录下的 `deploy/cpp` 文件夹，执行以下命令进行构建：

```bash
cd <OpenStereoROOT>/deploy/cpp
mkdir build && cd build
```

使用系统库中的 TensorRT 构建：

```bash
cmake .. && make
```

或者使用下载的 TensorRT tar 包进行构建：

```bash
cmake -DTENSORRT_ROOT=<path_to_tensorrt> .. && make
```

然后，可以使用以下命令进行推理：

```bash
./main <cfg_path> <engine_path> <left_image_path> <right_image_path> <options>
```

部署所需的头文件和动态链接库已打包在 `build/libopenstereo` 中，可方便地将其加入到你的工程中。

注意：使用 TensorRT tar 包进行构建时，请确保正确设置了 `LD_LIBRARY_PATH`。可使用以下命令查看链接的 TensorRT 动态库是否正确：

```bash
ldd main | grep libnv*
```
