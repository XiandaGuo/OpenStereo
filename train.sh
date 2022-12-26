# # **************** For sceneflow ****************
# # Baseline
CUDA_VISIBLE_DEVICES=5,7 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs ./configs/baseline/baseline_PSMNet.yaml --phase train

# # GaitSet
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs ./configs/gaitset/gaitset.yaml --phase train

# # GaitPart
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs ./configs/gaitpart/gaitpart.yaml --phase train

# GaitGL
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs ./configs/gaitgl/gaitgl.yaml --phase train

# # GLN 
# # Phase 1
# CUDA_VISIBLE_DEVICES=2,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs ./configs/gln/gln_phase1.yaml --phase train
# # Phase 2
# CUDA_VISIBLE_DEVICES=2,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs ./configs/gln/gln_phase2.yaml --phase train


# # **************** For kitti15 ****************
# # Baseline
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 opengait/main.py --cfgs ./configs/psmnet/baseline_OUMVLP.yaml --phase train

# # GaitSet
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 opengait/main.py --cfgs ./configs/gaitset/gaitset_OUMVLP.yaml --phase train

# # GaitPart
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 opengait/main.py --cfgs ./configs/gaitpart/gaitpart_OUMVLP.yaml --phase train

# GaitGL
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 opengait/main.py --cfgs ./configs/gaitgl/gaitgl_OUMVLP.yaml --phase train



# - https://mirrors.ustc.edu.cn/anaconda/cloud/bioconda/
#  - https://mirrors.ustc.edu.cn/anaconda/cloud/bioconda/linux-64
#  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
#  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
#  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
#  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/