# Prepare CREStereo Dataset

CREStereo Dataset is a synthetic dataset that contains 200000 pairs of images and the corresponding dense disparity map. </br>

Dataset can be downloaded at the following website: https://github.com/megvii-research/CREStereo/blob/master/README.md

The directory structure should be:
```text
CREStereoData
├── hole
|   ├── 0003c2ad-2128-4916-b450-60c22826bc83_left.jpg
|   ├── 0003c2ad-2128-4916-b450-60c22826bc83_right.jpg
|   ├── 0003c2ad-2128-4916-b450-60c22826bc83_left.disp.png
|   ├── 0003c2ad-2128-4916-b450-60c22826bc83_right.disp.png
...
├── reflective
|   |
...
├── shapenet
|   |
...
├── tree
|   |
...

```

_Optionally you can write your own txt file and use all the parts of the dataset._ 

```bibtex
@inproceedings{li2022practical,
  title={Practical stereo matching via cascaded recurrent network with adaptive correlation},
  author={Li, Jiankun and Wang, Peisen and Xiong, Pengfei and Cai, Tao and Yan, Ziwei and Yang, Lei and Liu, Jiangyu and Fan, Haoqiang and Liu, Shuaicheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16263--16272},
  year={2022}
}
```