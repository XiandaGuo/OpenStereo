#!/usr/bin/env bash

# 获取命令执行结果：$(pwd)
# 获取环境变量 $PATH
# 获取定义的变量 targetpath=xxx  $targetpath
# ${}变量展开 $()命令替换

# 调试模式
set -x

# 获取输入参数
NGPUS=$1
PY_ARGS=${@:2}

# 清除传入的位置参数，已保存到PY_ARGS
#set --

# 获取可用端口
while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done

# GPU选择
#export CUDA_VISIBLE_DEVICES=4,5

# 复制文件并cd到指定目录
datename=$(date +%Y%m%d-%H%M%S)
targetpath='chenming.zhang/workspace/LightStereo-'$datename
# 假如源目录写为/var/www/就会把该目录下所有文件同步到目标目录，
# 如果写为/var/www/*，那么当前目录下的隐藏文件则不会被同步，不过子目录中的隐藏文件还是会被同步
rsync -r --exclude='output' ./* $targetpath
cd ${targetpath}
echo $(pwd)

# 激活python环境
#source chenming.zhang/miniconda3/bin/activate
source chenming.zhang/miniconda3/etc/profile.d/conda.sh
conda activate stereo

# 执行命令
torchrun \
    --nnodes=1 \
    --nproc_per_node=${NGPUS} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:${PORT} \
    train.py --dist_mode ${PY_ARGS}

# 执行命令
python -u tools/train.py \
    --dist_mode \
    --cfg_file cfgs/igev/igev_stereo_nogru.yaml \
    --fix_random_seed \
    --workers 8 \
    --pin_memory \
    --eval_interval 1 \
    $PY_ARGS


# source chenming.zhang/miniconda3/bin/activate stereo &&
# cd chenming.zhang/code/LightStereo &&
# bash tools/scripts/train.sh --extra_tag
