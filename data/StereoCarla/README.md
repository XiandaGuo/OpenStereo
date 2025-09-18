# Prepare StereoCarla Dataset

StereoCarla is a high-fidelity synthetic stereo dataset specifically designed for autonomous driving scenarios. Built on the CARLA simulator, StereoCarla incorporates a wide range of camera configurations—including diverse baselines, viewpoints, and sensor placements—as well as varied environmental conditions such as lighting changes, weather effects, and road geometries. </br>

The dataset can be downloaded at the following website: https://xiandaguo.net/StereoCarla/

The directory structure should be:
```
StereoCarla
└───normal
    └───town01
         └───left
         └───baseline_010
             └───rgb
             └───depth
             └───pose
'''
    └───town10
└───pitch00
└───pitch30
└───roll05
└───roll15
└───roll30
└───weather

```

_Optionally, you can write your own txt file and use all the parts of the dataset._ 

```bibtex
@article{guo2025stereocarla,
      title={StereoCarla: A High-Fidelity Driving Dataset for Generalizable Stereo}, 
      author={Xianda Guo and Chenming Zhang and Ruilin Wang and Youmin Zhang and Wenzhao Zheng and Matteo Poggi and Hao Zhao and Qin Zou and Long Chen},
      year={2025},
      journal={arXiv preprint arXiv:2509.12683}
}
```
