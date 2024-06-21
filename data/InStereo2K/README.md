# Prepare InStereo2K Dataset

InStereo2K Dataset contains 2050 pairs of images with high accuracy disparity maps (2000 for training, 50 for testing), which can be downloaded at the following website: https://github.com/YuhuaXu/StereoDataset

The directory structure should be:
```text
InStereo2K
├── train
|   ├── part1
|   |   ├── 000000
|   |   ├── 000001
...
|   ├── part2
...
├── test
|   ├── 000040
|   ├── 000080
...
```

_Optionally you can write your own txt file and use all the parts of the dataset._ 

```bibtex
@article{bao2020instereo2k,
  title={Instereo2k: a large real dataset for stereo matching in indoor scenes},
  author={Bao, Wei and Wang, Wei and Xu, Yuhua and Guo, Yulan and Hong, Siyu and Zhang, Xiaohu},
  journal={Science China Information Sciences},
  volume={63},
  pages={1--11},
  year={2020},
  publisher={Springer}
}
```