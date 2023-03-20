import torch


def convert_weights(ori_w, open_w, new_w):
    ori_model_weights = torch.load(ori_w)['state_dict']
    open_w = torch.load(open_w)
    open_model_weights = open_w['model']

    converted_weights = {}

    for (k1, v1), (k2, v2) in zip(ori_model_weights.items(), open_model_weights.items()):
        if v1.shape == v2.shape:
            print(f"Convert {k1} -> {k2}")
            converted_weights[k2] = v1
        else:
            print(f"k1: {v1.shape}, k2: {v2.shape} not match!")
            print(f"ERROR: Skip {k1} -> {k2}")

    open_w['model'] = converted_weights
    torch.save(open_w, new_w)

    print(f"Save converted weights to {new_w}")


if __name__ == '__main__':
    ori_w = '/home/ralph/Downloads/epoch_10.pth'
    open_w = '/home/ralph/Projects/PhiGent/OpenStereo/output/SceneFlow/PSMNet/PSMNet_SceneFlow/checkpoints/PSMNet_SceneFlow-00500.pt'
    new_w = '/home/ralph/Projects/PhiGent/OpenStereo/output/SceneFlow/PSMNet/PSMNet_SceneFlow/checkpoints/PSMNet_SceneFlow-00501.pt'
    convert_weights(ori_w, open_w, new_w)
