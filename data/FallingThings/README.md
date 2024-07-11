# Prepare Falling Things Dataset

Falling Things Dataset is a synthetic dataset presented by NVIDIA. </br>

The dataset contains 60k annotated photos of 21 household objects taken from the YCB dataset.For each image, it contains the 3D poses, per-pixel class segmentation, and 2D/3D bounding box coordinates for all objects.The dataset provide mono and stereo RGB images, along with registered dense depth images.

Dataset can be downloaded at the following website: https://research.nvidia.com/publication/2018-06_falling-things-synthetic-dataset-3d-object-detection-and-pose-estimation

The directory structure should be:
```text
FallingThings
├── fat
|   ├── mixed
|   |   ├── kitchen_0
|   |   ├── kitchen_1
|   |   ├── kitchen_2
...
|   ├── single
|   |   ├── 002_master_chef_can_16k
|   |   ├── 003_cracker_box_16k

...

```

_Optionally you can write your own txt file and use all the parts of the dataset._ 

```bibtex
@inproceedings{tremblay2018falling,
  title={Falling things: A synthetic dataset for 3d object detection and pose estimation},
  author={Tremblay, Jonathan and To, Thang and Birchfield, Stan},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  pages={2038--2041},
  year={2018}
}
```