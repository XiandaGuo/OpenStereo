# Prepare MPI-Sintel Stereo Training Dataset

Sintel Dataset contains 1064 pairs of images with disparity maps. </br>
For each of the training sequences, it contains:
- The "clean" and "final" passes rendered from two cameras with a baseline of
  10 cm apart.
- The negative disparities.
- A mask showing the occluded pixels (which are visible in the left frame, but occluded in the right frame).
- A mask showing the out-of-frame pixels, which are visible in the left frame, but leave the visible area in the right frame. These pixels are not taken into account for the evaluation.

Dataset can be downloaded at the following website: http://sintel.is.tue.mpg.de/stereo

The directory structure should be:
```text
Sintel
├── training
|   ├── clean_left
|   |   ├── alley_1
|   |   ├── alley_2
...
|   ├── clean_right
|   |   ├── alley_1
|   |   ├── alley_2
...
|   ├── disparities
|   |   ├── alley_1
|   |   ├── alley_2
...
|   ├── final_left
|   |   ├── alley_1
|   |   ├── alley_2
...
|   ├── final_right
|   |   ├── alley_1
|   |   ├── alley_2
...
|   ├── occlusions
|   |   ├── alley_1
|   |   ├── alley_2
...

```

_Optionally you can write your own txt file and use all the parts of the dataset._ 

```bibtex
@inproceedings{Butler:ECCV:2012,
title = {A naturalistic open source movie for optical flow evaluation},
author = {Butler, D. J. and Wulff, J. and Stanley, G. B. and Black, M. J.},
booktitle = {European Conf. on Computer Vision (ECCV)},
editor = {{A. Fitzgibbon et al. (Eds.)}},
publisher = {Springer-Verlag},
series = {Part IV, LNCS 7577},
month = oct,
pages = {611--625},
year = {2012}
}
```