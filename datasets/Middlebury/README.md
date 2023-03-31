# Prepare Middlebury Stereo Datasets

This dataset was created by the Middlebury Computer Vision Lab for evaluating the performance of stereo matching
algorithms. The dataset includes multiple sets of stereo images and corresponding disparity images, which can be
downloaded at the following website: http://vision.middlebury.edu/stereo/data

The directory structure should be as follows:

```text
data
├── Middlebury
|   ├── MiddEval3
|   |   ├── TrainingQ
|   |   ├── TrainingH
|   |   ├── TrainingF
|   |   ├── TestingQ
|   |   ├── TestingH
|   |   ├── TestingF
...
```

_Optionally you can write your own txt file and use all the parts of the dataset._

### Reference

```bibtex
@inproceedings{scharstein2014high,
  title={High-resolution stereo datasets with subpixel-accurate ground truth},
  author={Scharstein, Daniel and Hirschm{\"u}ller, Heiko and Kitajima, York and Krathwohl, Greg and Ne{\v{s}}i{\'c}, Nera and Wang, Xi and Westling, Porter},
  booktitle={Pattern Recognition: 36th German Conference, GCPR 2014, M{\"u}nster, Germany, September 2-5, 2014, Proceedings 36},
  pages={31--42},
  year={2014},
  organization={Springer}
}
```