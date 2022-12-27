# Prepare KITTI 2012 dataset

KITTI stereo dataset is available at the KITTI [official website](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo).
Please download the dataset and place it in the `datasets/kitti12` directory.
The directory structure should be as follows:

```text
datasets
|   kitti12
|   |   ├── training
|   |   |   ├── colmap
|   |   |   |   ├── 000000
|   |   |   |   |   ├── 000000.png
|   |   |   |   |   ├── 000001.png
|   |   |   |   |   ├── 000002.png
...
|   |   ├── testing
|   |   |   ├── colmap
|   |   |   |   ├── 000000
|   |   |   |   |   ├── 000000.png
|   |   |   |   |   ├── 000001.png
|   |   |   |   |   ├── 000002.png
...
|   |   ├── train.txt
|   |   ├── val.txt
```

### Reference

```bibtex
@ARTICLE{Menze2018JPRS,
  author = {Moritz Menze and Christian Heipke and Andreas Geiger},
  title = {Object Scene Flow},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing (JPRS)},
  year = {2018}
}

@INPROCEEDINGS{Menze2015ISA,
  author = {Moritz Menze and Christian Heipke and Andreas Geiger},
  title = {Joint 3D Estimation of Vehicles and Scene Flow},
  booktitle = {ISPRS Workshop on Image Sequence Analysis (ISA)},
  year = {2015}
}
```