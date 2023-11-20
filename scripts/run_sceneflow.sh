# train
export CUDA_VISIBLE_DEVICES=0,1,2,3
python openstereo/main.py --config ./configs/igev/igev_sceneflow.yaml --scope train

# val
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#python openstereo/main.py --config ./configs/igev/igev_sceneflow.yaml --scope val
