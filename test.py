import os
import tqdm
import random

lines = open('datasets/sceneflow/train.txt').readlines()

random.shuffle(lines)

with open('datasets/sceneflow/train.txt', 'w') as f:
    f.writelines(lines[:int(len(lines)*0.8)])

with open('datasets/sceneflow/val.txt', 'w') as f:
    f.writelines(lines[int(len(lines)*0.8):])


# for line in tqdm.tqdm(lines):
#     line = line.strip()
#     img1, img2, disp = line.split(' ')
#     img1 = os.path.join('data/SceneFlow', img1)
#     img2 = os.path.join('data/SceneFlow', img2)
#     disp = os.path.join('data/SceneFlow', disp)
#     if not os.path.exists(img1):
#         print(img1)
#     if not os.path.exists(img2):
#         print(img2)
#     if not os.path.exists(disp):
#         print(disp)