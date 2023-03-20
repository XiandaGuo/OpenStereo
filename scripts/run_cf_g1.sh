cd $(readlink -f `dirname $0`)
export TORCH_HOME=/mnt/cfs/algorithm/youmin.zhang/torch_home

source /home/youmin.zhang/.bashrc

conda activate mmdet3d

cd /mnt/cfs/algorithm/youmin.zhang/juntao/OpenStereo

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 openstereo/main.py --cfgs ./configs/cfnet/CFNet_sceneflow_g1.yaml --phase train 2>&1 |tee logs/CFNet_sceneflow_train_g1.txt