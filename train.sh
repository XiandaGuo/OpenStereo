### GwcNet
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 openstereo/main.py --cfgs ./configs/gwcnet/GwcNet_sceneflow.yaml --phase train
