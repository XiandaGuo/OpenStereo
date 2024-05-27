#!/usr/bin/env bash

set -x

# python代码执行参数
NNODES=$1
PY_ARGS=${@:2}

# 环境变量
export TORCH_HOME='/mnt/nas/algorithm/chenming.zhang/.cache/torch'

# 激活python环境
set +x
source /mnt/nas/algorithm/chenming.zhang/miniconda3/etc/profile.d/conda.sh
conda activate stereo
set -x

# 获取master地址
master_name=$MASTER_ADDR
if [ $master_name == "localhost" ]; then
    master_name=$HOSTNAME
fi
master_addr=$(getent hosts "$master_name" | awk '{ print $1 }' | head -n1)
master_port=$MASTER_PORT

# 执行
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$master_addr:$master_port \
    tools/train.py \
        --dist_mode \
        --fix_random_seed \
        --root_dir '/mnt/nas/algorithm/chenming.zhang/github/OpenStereo/output' \
        --workers 8 \
        --pin_memory $PY_ARGS