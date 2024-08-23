# Prepare Argoverse Stereo Dataset

Argoverse 1 Stereo dataset consists of rectified stereo images and ground truth disparity maps for 74 out of the 113 Argoverse 1 3D Tracking Sequences. The stereo images are (2056 x 2464 px) and sampled at 5 Hz. </br>

Dataset can be downloaded at the following website: https://www.argoverse.org/av1.html#stereo-link

The directory structure should be:
```text
argoverse_stereo_v1.1
└───disparity_maps_v1.1
|   └───test
|   └───train
|   |    └───273c1883-673a-36bf-b124-88311b1a80be
|   |        └───stereo_front_left_rect_disparity
|   |        └───stereo_front_left_rect_objects_disparity
|   └───val
└───rectified_stereo_images_v1.1
    └───test
    └───train
    |    └───273c1883-673a-36bf-b124-88311b1a80be
    |        └───stereo_front_left_rect
    |        └───stereo_front_right_rect
    |            vehicle_calibration_stereo_info.json
    └───val

```

_Optionally you can write your own txt file and use all the parts of the dataset._ 

```bibtex
@inproceedings{chang2019argoverse,
  title={Argoverse: 3d tracking and forecasting with rich maps},
  author={Chang, Ming-Fang and Lambert, John and Sangkloy, Patsorn and Singh, Jagjeet and Bak, Slawomir and Hartnett, Andrew and Wang, De and Carr, Peter and Lucey, Simon and Ramanan, Deva and others},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={8748--8757},
  year={2019}
}
```