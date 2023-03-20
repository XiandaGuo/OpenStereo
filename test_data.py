import os


def find_missing_files(listfile, root_dir):
    missing_files = []
    # open SceneFlow listfile
    with open(listfile) as f:
        train_list = f.readlines()

    for line in train_list:
        line = line.strip()
        imgL, imgR, disp = line.split(" ")

        imgL = os.path.join(root_dir, imgL)
        imgR = os.path.join(root_dir, imgR)
        disp = os.path.join(root_dir, disp)

        if not os.path.exists(imgL):
            print("imgL not found: {}".format(imgL))
            missing_files.append(imgL)
        if not os.path.exists(imgR):
            print("imgR not found: {}".format(imgR))
            missing_files.append(imgR)
        if not os.path.exists(disp):
            print("disp not found: {}".format(disp))
            missing_files.append(disp)
        print('passed: {}'.format(line))

    print("total error: {}".format(len(missing_files)))
    return missing_files


def main():
    listfile = "datasets/SceneFlow/sceneflow_cleanpass_train.txt"
    root_dir = "./data/SceneFlow"
    missing_files = find_missing_files(listfile, root_dir)
    if len(missing_files) > 0:
        with open('./missing_files.txt', 'w') as f:
            for line in missing_files:
                f.write(line + '\n')


if __name__ == '__main__':
    main()
