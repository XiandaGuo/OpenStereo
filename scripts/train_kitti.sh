# CFNet sceneflow train
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 openstereo/main.py --cfgs ./configs/cfnet/CFNet_kitti15_g2.yaml --phase train 2>&1 |tee logs/CFNet_kitti15_train_g2.txt
