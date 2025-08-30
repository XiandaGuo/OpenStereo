## 1. Environment Installation

```
1. conda env create -f [environment.yml](https://github.com/NVlabs/FoundationStereo/blob/master/environment.yml)
2. conda run -n foundation_stereo pip install flash-attn
3. conda activate foundation_stereo
```

## 2. Training on SceneFlow Dataset

```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:23456 tools/train.py --dist_mode --cfg_file cfgs/foundationstereo/foundationstereo_sceneflow
```

## 3. Evaluation

```
python tools/eval.py --cfg_file cfgs/foundationstereo/foundationstereo_sceneflow --eval_data_cfg_file cfgs/sceneflow_eval.yaml --pretrained_model your_pretrained_ckpt_path
```

## Our Reproduced Results 

|                         Model                          |         Original Paper |    Ours|     Configuration | 
|:------------------------------------:|:---------------------:|------------------------:|:------------:|
| [FoundationStereo](https://arxiv.org/abs/2501.09898) |     0.33| **0.34**|        [foundationstereo_sceneflow.yaml](../cfgs/foundationstereo/fstereo_sceneflow.yaml)    | 

Access our checkpoint: [BaiduDrive](https://pan.baidu.com/s/1vA6xp9UMGJ3_tUahBrzIcw?pwd=mx7v) or [Google Drive](https://drive.google.com/drive/folders/1f1NrVMHUQqgqBA7Q5Q-pyZB65GNGBkHG?usp=drive_link)
