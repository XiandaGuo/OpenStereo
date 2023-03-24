
export CUDA_VISIBLE_DEVICES=0,1,2,3
python openstereo/main.py --config ./configs/fadnet/fadnet_sceneflow.yaml --scope train 