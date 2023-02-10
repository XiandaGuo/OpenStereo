### GwcNet
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 openstereo/main.py --cfgs ./configs/gwcnet/GwcNet_sceneflow.yaml --phase train

### CFNet
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 openstereo/main.py --cfgs ./configs/cfnet/CFNet_sceneflow.yaml --phase train

### GaNet
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 openstereo/main.py --cfgs ./configs/ganet/GaNet_sceneflow.yaml --phase train
