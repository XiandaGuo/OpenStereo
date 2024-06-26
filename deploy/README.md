python deploy/export.py --config cfgs/psmnet/psmnet_kitti15.yaml --weights output/KittiDataset/PSMNet/psmnet_kitti15/default/ckpt/checkpoint_epoch_0.pth --device 0 --simplify --half --include onnx engine

bash deploy/trt_profile.sh --onnx output/KittiDataset/PSMNet/psmnet_kitti15/default/ckpt/checkpoint_epoch_0.onnx --fp16 --verbose
bash deploy/trt_profile.sh --loadEngine output/KittiDataset/PSMNet/psmnet_kitti15/default/ckpt/checkpoint_epoch_0.engine --fp16 --verbose

sudo apt-get install libopencv-dev
sudo apt-get install python3-opencv

python -m onnxsim input_model.onnx output_model.onnx --overwrite-input-shape "left_img:1,3,384,1248" "right_img:1,3,384,1248"
python -m onnxsim output/KittiDataset/PSMNet/psmnet_kitti15/default/ckpt/checkpoint_epoch_0.onnx output/KittiDataset/PSMNet/psmnet_kitti15/default/ckpt/checkpoint_epoch_0.onnx --input-shape "left_img:1,3,384,1248" "right_img:1,3,384,1248"

onnxsim input_model.onnx output_model.onnx --overwrite-input-shape "left_img:1,3,384,1248" "right_img:1,3,384,1248"
onnxsim output/KittiDataset/PSMNet/psmnet_kitti15/default/ckpt/checkpoint_epoch_0.onnx output/KittiDataset/PSMNet/psmnet_kitti15/default/ckpt/checkpoint_epoch_0.onnx --input-shape "left_img:1,3,384,1248" "right_img:1,3,384,1248"