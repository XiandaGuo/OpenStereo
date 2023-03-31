# Prepare KITTI 2015 dataset

KITTI stereo dataset is available at the KITTI [official website](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo).
Please download the dataset and place it in the `datasets/kitti15` directory.
The directory structure should be as follows:

```text
data
|   kitti15
|   |   ├── training
|   |   |   ├── disp_noc_0
|   |   |   ├── disp_noc_1
|   |   |   ├── disp_occ_0
|   |   |   ├── disp_occ_1
|   |   |   ├── flow_noc
|   |   |   ├── flow_occ
|   |   |   ├── image_2
|   |   |   ├── image_3
|   |   |   ├── obj_map
|   |   |   ├── viz_flow_occ
|   |   |   ├── viz_flow_occ_dilate_1
...
|   |   ├── testing
|   |   |   ├── image_2
|   |   |   ├── image_3
...
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