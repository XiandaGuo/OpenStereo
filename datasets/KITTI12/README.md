# Prepare KITTI 2012 dataset

KITTI stereo dataset is available at the KITTI [official website](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo).
Please download the dataset and place it in the `datasets/kitti12` directory.
The directory structure should be as follows:

```text
data
|   kitti12
|   |   ├── calib
|   |   |   ├── calib
|   |   |   ├── colored_0
|   |   |   ├── colored_1
|   |   |   ├── disp_noc
|   |   |   ├── disp_occ
|   |   |   ├── disp_refl_noc
|   |   |   ├── disp_refl_occ
|   |   |   ├── flow_noc
|   |   |   ├── flow_occ
|   |   |   ├── image_0
|   |   |   ├── image_1
...
|   |   ├── testing
|   |   |   ├── calib
|   |   |   ├── colored_0
|   |   |   ├── colored_1
|   |   |   ├── image_0
|   |   |   ├── image_1
...
```

### Reference

```bibtex
@inproceedings{kitti2012,
  title={Are we ready for autonomous driving? the kitti vision benchmark suite},
  author={Geiger, Andreas and Lenz, Philip and Urtasun, Raquel},
  booktitle={CVPR},
  year={2012}
}
```
