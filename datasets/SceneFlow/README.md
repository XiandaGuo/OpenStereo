# Prepare SceneFlow dataset

Scene Flow Datasets is available at the Scene Flow [official website](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html).
Please download the following parts of FlyingThings3D, Driving, Monkaa and place it in the `datasets/sceneflow` directory.
- RGB images (finalpass)
- RGB images (cleanpass)
- Disparity

And for FlyingThings3D subset, you need to download the following parts:
- RGB images (cleanpass)
- Disparity
- Disparity Occlusions

The directory structure should be as follows:
```text
data
|   SceneFlow
|   ├── Driving
|   |   ├── frames_finalpass
|   |   ├── frames_cleanpass
|   |   ├── disparity
...
|   ├── Monkaa
|   |   ├── frames_finalpass
|   |   ├── frames_cleanpass
|   |   ├── disparity
...
|   ├── FlyingThings3D
|   |   ├── frames_finalpass
|   |   ├── frames_cleanpass
|   |   ├── disparity
...
|   ├── FlyingThings3D_subset
|   |   ├── train
|   |   |   ├── frames_cleanpass
|   |   |   ├── disparity
|   |   |   ├── disparity_occlusions
|   |   ├── test
|   |   |   ├── frames_cleanpass
|   |   |   ├── disparity
|   |   |   ├── disparity_occlusions
```


_Optionally you can write your own txt file and use all the parts of the dataset._ 

### Reference 

```bibtex
@inproceedings{mayer2016large,
  title={A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation},
  author={Mayer, Nikolaus and Ilg, Eddy and Hausser, Philip and Fischer, Philipp and Cremers, Daniel and Dosovitskiy, Alexey and Brox, Thomas},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4040--4048},
  year={2016}
}
```