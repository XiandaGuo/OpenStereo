from torch.utils.data import Dataset

from . import stereo_trans as ST


class StereoDataset(Dataset):
    def __init__(self, data_cfg, scope='train'):
        self.data_cfg = data_cfg
        self.is_train = scope == 'train'
        self.scope = scope.lower()
        self.dataset = None
        self.transform = None
        self.build_dataset()

    def build_dataset(self):
        if self.data_cfg['name'] in ['KITTI2012', 'KITTI2015']:
            from data.reader.kitti_reader import KittiReader
            self.dataset = KittiReader(self.data_cfg['root'], self.data_cfg[f'{self.scope}_list'])
        elif self.data_cfg['name'] == 'SceneFlow':
            from data.reader.sceneflow_reader import SceneFlowReader
            self.dataset = SceneFlowReader(self.data_cfg['root'], self.data_cfg[f'{self.scope}_list'])
        elif self.data_cfg['name'] == 'Middlebury':
            from data.reader.middlebury_reader import MiddleburyReader
            self.dataset = MiddleburyReader(self.data_cfg['root'], self.data_cfg[f'{self.scope}_list'])
        elif self.data_cfg['name'] == 'ETH3D':
            from data.reader.eth3d_reader import ETH3DReader
            self.dataset = ETH3DReader(self.data_cfg['root'], self.data_cfg[f'{self.scope}_list'])
        else:
            name = self.data_cfg['name']
            raise NotImplementedError(f'{name} dataset is not supported yet.')
        self.build_transform()

    def build_transform(self):
        transform_config = self.data_cfg['transform']
        config = transform_config['train'] if self.is_train else transform_config['test']
        size, mean, std = config['size'], config['mean'], config['std']
        if self.is_train:
            transform = ST.Compose([
                # ST.RandomHorizontalFlip(),
                ST.RandomCrop(size),
                # ST.GetValidDispNOcc(),
                ST.GetValidDisp(192),
                ST.TransposeImage(),
                ST.ToTensor(),
                ST.NormalizeImage(mean, std),
            ])
        else:
            transform = ST.Compose([
                ST.StereoPad(size),
                # ST.GetValidDispNOcc(),
                ST.GetValidDisp(192),
                ST.TransposeImage(),
                ST.ToTensor(),
                ST.NormalizeImage(mean, std),
            ])

        self.transform = transform

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.dataset)
