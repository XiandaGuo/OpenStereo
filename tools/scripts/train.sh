#!/usr/bin/env bash

set -x

# python代码执行参数
PY_ARGS=${@:1}

# 环境变量
export TORCH_HOME='/mnt/nas/algorithm/chenming.zhang/.cache/torch'

# 复制文件并cd到指定目录
datename=$(date +%Y%m%d-%H%M%S)
targetpath='/mnt/nas/algorithm/chenming.zhang/workspace/LightStereo-'$datename
rsync -r --exclude='output' ./* $targetpath
cd $targetpath
# echo $(pwd)

# 获取可用端口
while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "$status" != "0" ]; then
        break;
    fi
done

# 激活python环境
set +x
source /mnt/nas/algorithm/chenming.zhang/miniconda3/etc/profile.d/conda.sh
conda activate stereo
set -x

# 执行
torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:$PORT \
    tools/train.py \
        --dist_mode \
        --fix_random_seed \
        --save_root_dir '/mnt/nas/algorithm/chenming.zhang/code/LightStereo/output' \
        --pin_memory $PY_ARGS
