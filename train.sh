### GwcNet
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 openstereo/main.py --cfgs ./configs/gwcnet/GwcNet_kitti12.yaml --phase train
