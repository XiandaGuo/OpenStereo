cd $(readlink -f `dirname $0`)
export TORCH_HOME=/mnt/cfs/algorithm/youmin.zhang/torch_home

source /home/youmin.zhang/.bashrc

conda activate mmdet3d

cd /mnt/cfs/algorithm/youmin.zhang/juntao/OpenStereo

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --batchSize=1 \
#                 --crop_height=240 \
#                 --crop_width=528 \
#                 --max_disp=192 \
#                 --thread=4 \
#                 --data_path='/mnt/cfs/algorithm/xianda.guo/data/stereo/Kitti2012/training/' \
#                 --training_list='lists/kitti2012_train.list' \
#                 --save_path='./checkpoint/finetune_kitti' \
#                 --kitti=1 \
#                 --shift=3 \
#                 # --resume='./checkpoint/sceneflow_epoch_10.pth' \
#                 --nEpochs=800 2>&1 |tee logs/log_finetune2_kitti.txt


# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 openstereo/main.py \ 
#                 --cfgs ./configs/ganet/GaNet_sceneflow.yaml --phase train \
#                 2>&1 |tee logs/GaNet_sceneflow.txt

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 openstereo/main.py --cfgs ./configs/gwcnet/GwcNet_sceneflow.yaml --phase train 2>&1 |tee logs/GwcNet_sceneflow.txt