import os
from natsort import natsorted

target_dir = '/mnt/nas/algorithm/zixuan.li/yiqun.duan/OpenStereo/datasets/SceneFlow/FlyingThings3D_sttr_test.txt'
sttr_path = '/mnt/nas/algorithm/xianda.guo/data/FlyingThings3D/'

def generate_dataset_txt(datadir, split_folder):
    # 处理frame_finalpass文件夹
    frame_directory = os.path.join(datadir, 'frame_finalpass', split_folder)
    sub_folders = [os.path.join(frame_directory, subset) for subset in os.listdir(frame_directory) if
                   os.path.isdir(os.path.join(frame_directory, subset))]

    seq_folders = []
    for sub_folder in sub_folders:
        seq_folders += [os.path.join(sub_folder, seq) for seq in os.listdir(sub_folder) if
                        os.path.isdir(os.path.join(sub_folder, seq))]

    left_data = []
    for seq_folder in seq_folders:
        left_folder = os.path.join(seq_folder, 'left')
        left_data += [os.path.join(left_folder, img) for img in os.listdir(left_folder)]

    # 自然排序
    left_data = natsorted(left_data)

    # 处理occlusion文件夹
    occlusion_directory = os.path.join(datadir, 'occlusion', split_folder, 'left')
    occ_data = [os.path.join(occlusion_directory, occ) for occ in os.listdir(occlusion_directory)]
    occ_data = natsorted(occ_data)

    # 生成文本文件
    with open(target_dir, 'w') as f:
        for idx, left_path in enumerate(left_data):
            right_path = left_path.replace('left', 'right')
            occ_left_path = occ_data[idx]
            occ_right_path = occ_left_path.replace('left', 'right')
            disp_left_path = left_path.replace('frame_finalpass', 'disparity').replace('.png', '.pfm')
            disp_right_path = right_path.replace('frame_finalpass', 'disparity').replace('.png', '.pfm')

            line = f"{left_path} {right_path} {disp_left_path} {disp_right_path} {occ_left_path} {occ_right_path}\n".replace(sttr_path, '')
            f.write(line)

# Example usage
generate_dataset_txt(sttr_path, 'TEST')

