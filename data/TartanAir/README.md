# Prepare TartanAir Dataset

The TartanAir is for robot navigation task and more. The data is collected in photo-realistic simulation environments in the presence of various light conditions, weather and moving objects. By collecting data in simulation, you are able to obtain multi-modal sensor data and precise ground truth labels, including the stereo RGB image, depth image, segmentation, optical flow, camera poses, and LiDAR point cloud. It set up a large number of environments with various styles and scenes, covering challenging viewpoints and diverse motion patterns, which are difficult to achieve by using physical data collection platforms. 
 </br>

Dataset can be downloaded at the following website: https://theairlab.org/tartanair-dataset/

The directory structure should be:
```text
data
└───env01
|     └───Easy
|     |     └───P001
|     |     |      └───depth_left
|     |     |      |        └───000000_left_depth.npy
|     |     |      |        ...
|     |     |      └───depth_right
|     |     |      |        └───000000_right_depth.npy 000000_left_seg.npy
|     |     |      |        ...
|     |     |      └───flow
|     |     |      |     └───000000_000001_flow.npy
|     |     |      |     └───000000_000001_mask.npy
|     |     |      |        ...
|     |     |      └───image_left
|     |     |      |        └───000000_left.png
|     |     |      |        ...
|     |     |      └───image_right
|     |     |      |        └───000000_firht.png
|     |     |      |        ...
|     |     |      └───seg_left
|     |     |      |        └───000000_left_seg.npy
|     |     |      |        ...
|     |     |      └───seg_right
|     |     |      |        └───000000_right_seg.npy
|     |     |      |        ...
|     |     |      └───pose_left.txt
|     |     |      └───pose_right.txt
|     |     ...
|     └───Hard
|     |     └───P001
|     |     |      └───depth_left
|     |     |      |        └───000000_left_depth.npy
|     |     |      |        ...
|     |     |      └───depth_right
|     |     |      |        └───000000_right_depth.npy 000000_left_seg.npy
|     |     |      |        ...
|     |     |      └───flow
|     |     |      |     └───000000_000001_flow.npy
|     |     |      |     └───000000_000001_mask.npy
|     |     |      |        ...
|     |     |      └───image_left
|     |     |      |        └───000000_left.png
|     |     |      |        ...
|     |     |      └───image_right
|     |     |      |        └───000000_firht.png
|     |     |      |        ...
|     |     |      └───seg_left
|     |     |      |        └───000000_left_seg.npy
|     |     |      |        ...
|     |     |      └───seg_right
|     |     |      |        └───000000_right_seg.npy
|     |     |      |        ...
|     |     |      └───pose_left.txt
|     |     |      └───pose_right.txt
|     |     ...
...
```
_Optionally you can write your own txt file and use all the parts of the dataset._ 

```bibtex
@article{tartanair2020iros,
  title =   {TartanAir: A Dataset to Push the Limits of Visual SLAM},
  author =  {Wang, Wenshan and Zhu, Delong and Wang, Xiangwei and Hu, Yaoyu and Qiu, Yuheng and Wang, Chen and Hu, Yafei and Kapoor, Ashish and Scherer, Sebastian},
  booktitle = {2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year =    {2020}
}
```

