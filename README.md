<img src="./misc/OpenStereo.png" alt="logo" />

------------------------------------------
OpenStereo is a flexible and extensible project for stereo matching.

## What's New
- **[March 2023]**:OpenStereo is available!!!

## Highlighted features
- **Mutiple Dataset supported**: OpenStereo supports six popular stereo datasets: [SceneFlow](datasets/SceneFlow/README.md), [KITTI12](datasets/KITTI12/README.md) & [KITTI15](datasets/KITTI15/README.md), 
 [ETH3D](datasets/ETH3D/README.md),[Middlebury](datasets/Middlebury/README.md) and [DrivingStereo](datasets/DrivingStereo/README.md) .
- **Multiple Models Support**: We reproduced several SOTA methods, and reached the same or even the better performance. 
- **DDP Support**: The officially recommended [`Distributed Data Parallel (DDP)`](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) mode is used during both the training and testing phases.
- **AMP Support**: The [`Auto Mixed Precision (AMP)`](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html?highlight=amp) option is available.
- **Nice log**: We use [`tensorboard`](https://pytorch.org/docs/stable/tensorboard.html) and `logging` to log everything, which looks pretty.


## Getting Started

Please see [0.get_started.md](docs/0.get_started.md). We also provide the following tutorials for your reference:
- [Prepare dataset](docs/2.prepare_dataset.md)
- [Detailed configuration](docs/3.detailed_config.md)
- [Customize model](docs/4.how_to_create_your_model.md)
- [Advanced usages](docs/5.advanced_usages.md) 

## Model Zoo
Results and models are available in the [model zoo](docs/1.model_zoo.md).


## Authors:
**OpenStereo Team (OST)**
- [Xianda Guo (国显达)](https://scholar.google.com.hk/citations?hl=zh-CN&user=jPvOqgYAAAAJ), xianda_guo@163.com
- [Juntao Lu (陆俊陶)](https://github.com/ralph0813), juntao.lu@student.unimelb.edu.au
- [Yiqi Wang (王仪琦)](), wangyiqi18@mails.ucas.edu.cn
- [Yiqun Duan (段逸群)](https://github.com/duanyiqun), duanyiquncc@gmail.com
- [Zheng Zhu (朱政)](https://scholar.google.com.hk/citations?user=NmwjI0AAAAAJ&hl=zh-CN),zhengzhu@ieee.org


## Acknowledgement
- [AANet](https://github.com/haofeixu/aanet)
- [ACVNet](https://github.com/gangweiX/ACVNet)
- [CascadeStereo](https://github.com/alibaba/cascade-stereo)
- [CFNet](https://github.com/gallenszl/CFNet)
- [COEX](https://github.com/antabangun/coex)
- [DenseMatching](https://github.com/DeepMotionAIResearch/DenseMatchingBenchmark)
- [FADNet](https://github.com/HKBU-HPML/FADNet)
- [GANet](https://github.com/feihuzhang/GANet)
- [GwcNet](https://github.com/xy-guo/GwcNet)
- [LacGwcNet](https://github.com/SpadeLiu/Lac-GwcNet)
- [MSNet](https://github.com/cogsys-tuebingen/mobilestereonet)
- [PSMNet](https://github.com/JiaRenChang/PSMNet)
- [RAFT](https://github.com/princeton-vl/RAFT-Stereo)
- [STTR](https://github.com/mli0603/stereo-transformer)
- [OpenGait](https://github.com/ShiqiYu/OpenGait)
