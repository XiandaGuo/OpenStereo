# Prepare UnrealStereo4K Dataset

UnrealStereo4K Dataset contains 8200 pairs of images from 8 static scenes, including indoor and outdoor environments.. </br>

Dataset can be downloaded at the following website: https://github.com/fabiotosi92/SMD-Nets

The directory structure should be:
```text
UnrealStereo4K
├── 00000
|   ├── Disp0
|   |   ├── 00000.npy
|   |   ├── 00001.npy
...
|   ├── Disp1
|   |   ├── 00000.npy
|   |   ├── 00001.npy
...
|   ├── Image0
|   |   ├── 00000.png
|   |   ├── 00001.png
...
|   ├── Image1
|   |   ├── 00000.png
|   |   ├── 00001.png
├── 00001
|   |
...

```

_Optionally you can write your own txt file and use all the parts of the dataset._ 

```bibtex
@inproceedings{tosi2021smd,
  title={Smd-nets: Stereo mixture density networks},
  author={Tosi, Fabio and Liao, Yiyi and Schmitt, Carolin and Geiger, Andreas},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={8942--8952},
  year={2021}
}
```