
cd /file_system/vepfs/algorithm/xianda.guo/code/OpenStereo0620
source /file_system/vepfs/algorithm/xianda.guo/miniconda3/bin/activate fstereo
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:23456 tools/train.py --dist_mode  --save_root_dir /file_system/nas/algorithm/xianda.guo/checkpoint/OpenStereo/Output --extra_tag 0802 --cfg_file cfgs/foundationstereo/fstereo_dynamic.yaml

bash tools/scripts/eval_stereoanything.sh Output/MultiDataset/FoundationStereo/fstereo_mixdata/0815
bash tools/scripts/change_permission.sh Output/MultiDataset/FoundationStereo