# Prepare Spring Stereo Dataset

The Spring dataset is a computer-generated dataset for training and evaluating scene flow, optical flow and stereo methods. The dataset consists of stereoscopic video sequences and ground truth scene flow in its standard parametrization with reference frame disparity, target frame disparity and optical flow.
 </br>

Dataset can be downloaded at the following website: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-3376

The directory structure should be:
```text
spring
└───train
|   └───0001
|   |    └───frame_left
|   |    └───frame_right
|   |    └───disp1_left
|   |    └───disp1_right

|   ...
└───test
|   └───0001
|   |    └───frame_left
|   |    └───frame_right
```

_Optionally you can write your own txt file and use all the parts of the dataset._ 

```bibtex
@inproceedings{mehl2023spring,
  title={Spring: A high-resolution high-detail dataset and benchmark for scene flow, optical flow and stereo},
  author={Mehl, Lukas and Schmalfuss, Jenny and Jahedi, Azin and Nalivayko, Yaroslava and Bruhn, Andr{\'e}s},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4981--4991},
  year={2023}
}
```