# Prepare ETH3D Dataset

The ETH3D dataset is a large-scale dataset for visual odometry, dense reconstruction, and camera pose estimation
provided by the Computer Vision and Geometry Group at ETH Zurich. The dataset includes image sequences captured from
multiple cameras and corresponding camera poses, depth maps, and dense point clouds. The dataset can be used to evaluate
different types of algorithms, such as structure-based and direct-based visual odometry algorithms, dense reconstruction
algorithms, and camera pose estimation algorithms.

The dataset can be downloaded at the following website: https://www.eth3d.net/datasets#downloads

We only use the two-view stereo images and corresponding camera poses in the dataset. 
The directory structure should be:

```text
data
├── ETH3D
|   ├── two_view_training
|   |   ├── delivery_area_1l
...
|   ├── two_view_testing
|   |   ├── lakeside_1l
...
```

_Optionally you can write your own txt file and use all the parts of the dataset._

### Reference

```bibtex
@inproceedings{schoeps2017cvpr,
  author = {Thomas Sch\"ops and Johannes L. Sch\"onberger and Silvano Galliani and Torsten Sattler and Konrad Schindler and Marc Pollefeys and Andreas Geiger},
  title = {A Multi-View Stereo Benchmark with High-Resolution Images and Multi-Camera Videos},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2017}
}
```