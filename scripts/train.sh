# GwcNet sceneflow train
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 openstereo/main.py --cfgs ./configs/gwcnet/GwcNet_sceneflow_g1.yaml --phase train 2>&1 |tee logs/GwcNet_sceneflow_train_g1.txt
# CFNet sceneflow train
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 openstereo/main.py --cfgs ./configs/cfnet/CFNet_sceneflow_g1.yaml --phase train 2>&1 |tee logs/CFNet_sceneflow_train_g1.txt
# CasNet sceneflow train
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 openstereo/main.py --cfgs ./configs/casnet/CasNet_sceneflow_g1.yaml --phase train 2>&1 |tee logs/CasNet_sceneflow_train_g1.txt