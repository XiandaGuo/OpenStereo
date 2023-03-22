import numpy as np
import torch

from data.stereo_dataset_batch import StereoBatchDataset
from data.stereo_dataset import StereoDataset

if __name__ == '__main__':
    data_cfg = {
        'name': 'SceneFlow',
        'root': '/home/ralph/Projects/PhiGent/OpenStereo/data/SceneFlow/FlyingThings3D_subset',
        'train_list': '/home/ralph/Projects/PhiGent/OpenStereo/datasets/sceneflow/FlyingThings3D_subset_train.txt',
        'test_list': '/home/ralph/Projects/PhiGent/OpenStereo/datasets/sceneflow/FlyingThings3D_subset_train.txt',
        'transform': {
            'train': {
                'size': [256, 512],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            },
            'test': {
                'size': [256, 512],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }
    }

    mydata = StereoBatchDataset(
        data_cfg=data_cfg,
        scope='test'
    )
    batch_result = mydata[1, 2]

    for key in batch_result.keys():
        print(key, batch_result[key].shape)

    disp = batch_result['disp']
    disp_right = batch_result['disp_right']
    left = batch_result['left']
    right = batch_result['right']
    occ = batch_result['occ_mask']
    occ_right = batch_result['occ_mask_right']

    # display the disparity map
    import matplotlib.pyplot as plt

    nocc = ~occ[0].numpy()
    nocc_right = ~occ_right[0].numpy()
    n = np.concatenate([nocc, nocc_right], axis=0)
    # print(n)

    disp_l = disp[0].numpy() * nocc
    disp_r = disp_right[0].numpy() * nocc_right
    disp = np.concatenate([disp_l, disp_r], axis=0)
    left_right = np.concatenate([left[0].numpy(), right[0].numpy()], axis=1)
    left_right = np.transpose(left_right, (1, 2, 0))
    # undo the normalization
    left_right = left_right * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # RGB
    # img = np.concatenate([disp, left_right], axis=1)
    # plt.imshow(img)
    plt.subplot(1, 2, 1)
    plt.imshow(n)
    plt.subplot(1, 2, 2)
    plt.imshow(left_right)
    plt.show()

    # data_cfg = {
    #     'name': 'SceneFlow',
    #     'root': '/home/ralph/Projects/PhiGent/OpenStereo/data/SceneFlow/',
    #     'train_list': '/home/ralph/Projects/PhiGent/OpenStereo/datasets/sceneflow/debug_train.txt',
    #     'test_list': '/home/ralph/Projects/PhiGent/OpenStereo/datasets/sceneflow/denug_train.txt',
    #     'transform': {
    #         'train': {
    #             'size': [256, 512],
    #             'mean': [0.485, 0.456, 0.406],
    #             'std': [0.229, 0.224, 0.225]
    #         },
    #         'test': {
    #             'size': [384, 1248],
    #             'mean': [0.485, 0.456, 0.406],
    #             'std': [0.229, 0.224, 0.225]
    #         }
    #     }
    # }
    #
    # mydata = StereoDataset(
    #     data_cfg=data_cfg,
    #     scope='train'
    # )
    # for i in range(len(mydata)):
    #     batch_result = mydata[i]
    #
    #     # for key in batch_result.keys():
    #     #     print(key, batch_result[key].shape)
    #
    #     disp = batch_result['disp']
    #     disp_right = batch_result['disp_right']
    #     left = batch_result['left']
    #     right = batch_result['right']
    #     # occ = batch_result['occ_mask']
    #     # occ_right = batch_result['occ_mask_right']
    #
    #     # display the disparity map
    #     import matplotlib.pyplot as plt
    #
    #     # nocc = ~occ[0].numpy()
    #     # nocc_right = ~occ_right[0].numpy()
    #
    #     disp_l = disp.numpy()[0]
    #     disp_r = disp_right.numpy()[0]
    #     assert torch.all(disp[0] >= 0), 'disp should be non-negative'
    #     disp = np.concatenate([disp_l,disp_r], axis=0)
    #     left_right = np.concatenate([left.numpy(), right.numpy()], axis=1)
    #     left_right = np.transpose(left_right, (1, 2, 0))
    #     # undo the normalization
    #     left_right = left_right * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # RGB
    #
    #     # img = np.concatenate([disp, left_right], axis=1)
    #     # plt.imshow(img)
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(disp)
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(left_right)
    #     plt.show()
