# Prepare SceneFlow dataset

Scene Flow Datasets is available at the Scene Flow [official website](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html).
Please download the following parts of FlyingThings3D dataset and place it in the `datasets/sceneflow` directory.
- RGB images (finalpass)	
- Disparity

The directory structure should be as follows:
```text
datasets
|   sceneflow
|   |   ├── frames_finalpass
|   |   |   ├── TRAIN
|   |   |   |   ├── A
|   |   |   |   |   ├── 0000
|   |   |   |   |   |   ├── left
|   |   |   |   |   |   |   ├── 0006.png
|   |   |   |   |   |   |   ├── 0007.png
...
|   |   |   |   |   |   ├── right
|   |   |   |   |   |   |   ├── 0006.png
|   |   |   |   |   |   |   ├── 0007.png
...
|   |   |   |   |   |   ├── disparity
|   |   |   |   |   |   |   ├── 0006.pfm
|   |   |   |   |   |   |   ├── 0007.pfm
...
|   |   |   ├── TEST
|   |   |   |   ├── A
|   |   |   |   |   ├── 0000
|   |   |   |   |   |   ├── left
|   |   |   |   |   |   |   ├── 0006.png
|   |   |   |   |   |   |   ├── 0007.png
...
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