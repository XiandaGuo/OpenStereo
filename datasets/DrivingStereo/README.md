# Prepare Driving Stereo Dataset

Driving Stereo Datasets is available at the Driving Stereo [official website](https://drivingstereo-dataset.github.io/).
Please download the dataset from Google Drive or BaiduCloud and unzip it to the `data` directory.

To use our list file for training and testing, the directory structure should be as follows:
```text
data
├── DrivingStereo
|   ├── calib_test
|   ├── calib_train
|   ├── cloudy
|   ├── foggy
|   ├── rainy
|   ├── sunny
|   ├── test-depth-map
|   ├── test-disparity-map
|   ├── test-left-image
|   ├── test-right-image
|   ├── train-depth-map
|   ├── train-disparity-map
|   ├── train-left-image
|   ├── train-right-image
```

_Optionally you can write your own txt file and use all the parts of the dataset._ 

### Reference 

```bibtex
@inproceedings{yang2019drivingstereo,
    title={DrivingStereo: A Large-Scale Dataset for Stereo Matching in Autonomous Driving Scenarios},
    author={Yang, Guorun and Song, Xiao and Huang, Chaoqin and Deng, Zhidong and Shi, Jianping and Zhou, Bolei},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2019}
}
```