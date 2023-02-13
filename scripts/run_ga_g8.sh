cd $(readlink -f `dirname $0`)
export TORCH_HOME=/mnt/cfs/algorithm/youmin.zhang/torch_home

#source /home/youmin.zhang/.bashrc

conda activate mmdet3d

cd /mnt/cfs/algorithm/youmin.zhang/juntao/OpenStereo

cd openstereo && sh compile_libs.sh && cd ..

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
  --nproc_per_node=8 openstereo/main.py \
  --cfgs ./configs/ganet/GaNet_sceneflow_g8.yaml \
  --phase train \
  2>&1 |tee logs/GaNet_sceneflow_train_g8.txt


CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --nproc_per_node=2 openstereo/main.py \
  --cfgs ./configs/ganet/GaNet_sceneflow_g8.yaml \
  --phase train \
  2>&1 |tee logs/GaNet_sceneflow_train_g8.txt