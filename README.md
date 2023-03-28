## What's New

## Highlighted features
- **Mutiple Dataset supported**: OpenGait supports four popular stereo datasets: [sceneflow](?), [kitti12](?), and [kitti15](?).
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
**Open Stereo Team (OST)**
- [Xianda Guo (国显达)](https://scholar.google.com.hk/citations?hl=zh-CN&user=jPvOqgYAAAAJ), xianda_guo@163.com
- [Juntao Lu (陆俊陶)](https://github.com/ralph0813), juntao.lu@student.unimelb.edu.au
- [Yiqi Wang (王仪琦)](), wangyiqi18@mails.ucas.edu.cn
- [Yiqun Duan (段逸群)](https://github.com/duanyiqun), duanyiquncc@gmail.com



## Acknowledgement
- [OpenGait](https://github.com/ShiqiYu/OpenGait)
- [DenseMatching](https://github.com/DeepMotionAIResearch/DenseMatchingBenchmark)
- [AANet](https://github.com/haofeixu/aanet)
- [ACVNet](https://github.com/gangweiX/ACVNet)
- [CascadeStereo](https://github.com/alibaba/cascade-stereo)
- [CFNet](https://github.com/gallenszl/CFNet)
- [COEX](https://github.com/antabangun/coex)
- [FADNet](https://github.com/HKBU-HPML/FADNet)
- [GANet](https://github.com/feihuzhang/GANet)
- [GwcNet](https://github.com/xy-guo/GwcNet)
- [LacGwcNet](https://github.com/SpadeLiu/Lac-GwcNet)
- [MSNet](https://github.com/cogsys-tuebingen/mobilestereonet)
- [PSMNet](https://github.com/JiaRenChang/PSMNet)
- [RAFT](https://github.com/princeton-vl/RAFT-Stereo)
- [STTR](https://github.com/mli0603/stereo-transformer)
