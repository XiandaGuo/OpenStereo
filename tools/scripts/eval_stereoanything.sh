#!/bin/bash

source tools/scripts/constant.sh

dir_path="${CKPT_ROOT_DIR%/}/${1%/}/ckpt"

# 检查目录是否存在
if [ ! -d "$dir_path" ]; then
    echo "错误：目录 $dir_path 不存在"
    exit 1  # 非零退出码表示脚本异常终止
fi

pretrained_model=$(find "$dir_path" -maxdepth 1 -type f -name "checkpoint_epoch_*.pth" | \
        sed -n 's/.*epoch_\([0-9]*\)\.pth/\1 &/p' | \
        sort -n | \
        tail -n 1 | \
        cut -d' ' -f2)

echo -e "\033[32m$(printf '#%.0s' {1..50})\033[0mEval Checkpoint of SceneFlowDataset \033[32m$(printf '#%.0s' {1..50})\033[0m"
#pretrained_model='/file_system/vepfs/algorithm/ruilin.wang/code/LightStereoX/output/v2/Foundation.pt'
pretrained_model='/file_system/nas/algorithm/xianda.guo/checkpoint/OpenStereo/Output/SceneFlowDataset/FoundationStereo/foundationstereo_sceneflow/e50-lr0p001/ckpt/checkpoint_epoch_59.pth'

echo $pretrained_model

master_port=2335
cfg_file="cfgs/foundationstereo/fstereo_sceneflow.yaml"

nproc_per_node=4
export CUDA_VISIBLE_DEVICES=0,2,3,4

torchrun --nnodes=1 --nproc_per_node=4 --master_port=2335 \
tools/eval.py --dist_mode --cfg_file cfgs/foundationstereo/fstereo_sceneflow.yaml --pretrained_model $pretrained_model \
--eval_data_cfg_file cfgs/foundationstereo/sceneflow_eval.yaml
echo -e "\033[32m$(printf '#%.0s' {1..50})\033[0mKitti12 evaluated (see D1-all ↑)\033[32m$(printf '#%.0s' {1..50})\033[0m"
echo

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node --master_port=$master_port \
tools/eval.py --dist_mode --cfg_file $cfg_file --pretrained_model $pretrained_model \
--eval_data_cfg_file cfgs/foundationstereo/kitti12_eval.yaml
echo -e "\033[32m$(printf '#%.0s' {1..50})\033[0mKitti12 evaluated (see D1-all ↑)\033[32m$(printf '#%.0s' {1..50})\033[0m"
echo

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node --master_port=$master_port \
tools/eval.py --dist_mode --cfg_file $cfg_file --pretrained_model $pretrained_model \
--eval_data_cfg_file cfgs/foundationstereo/kitti12_eval.yaml
echo -e "\033[32m$(printf '#%.0s' {1..50})\033[0mKitti12 evaluated (see D1-all ↑)\033[32m$(printf '#%.0s' {1..50})\033[0m"
echo

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node --master_port=$master_port \
tools/eval.py --dist_mode --cfg_file $cfg_file --pretrained_model $pretrained_model \
--eval_data_cfg_file cfgs/foundationstereo/kitti15_eval.yaml
echo -e "\033[32m$(printf '#%.0s' {1..50})\033[0mKitti15 evaluated (see D1-all)\033[32m$(printf '#%.0s' {1..50})\033[0m"
echo

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node --master_port=$master_port \
tools/eval.py --dist_mode --cfg_file $cfg_file --pretrained_model $pretrained_model \
--eval_data_cfg_file cfgs/foundationstereo/middlebury_eval.yaml
echo -e "\033[32m$(printf '#%.0s' {1..50})\033[0mMidd evaluated (see Bad2.0)\033[32m$(printf '#%.0s' {1..50})\033[0m"
echo

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node --master_port=$master_port \
tools/eval.py --dist_mode --cfg_file $cfg_file --pretrained_model $pretrained_model \
--eval_data_cfg_file cfgs/foundationstereo/eth3d_eval.yaml
echo -e "\033[32m$(printf '#%.0s' {1..50})\033[0mETH3D evaluated (see Bad1.0)\033[32m$(printf '#%.0s' {1..50})\033[0m"
echo
