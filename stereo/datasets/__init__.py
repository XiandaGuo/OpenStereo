import torch
from functools import partial
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from .dataset_template import custom_collate

from .sceneflow_dataset import SceneFlowDataset
from .kitti_dataset import KittiDataset
from .driving_dataset import DrivingDataset
from .sceneflow_dataset import FlyingThings3DSubsetDataset
from .middlebury_dataset import MiddleburyDataset
from .eth3d_dataset import ETH3DDataset


__all__ = {
    'SceneFlowDataset': SceneFlowDataset,
    'KittiDataset': KittiDataset,
    'DrivingDataset': DrivingDataset,
    'FlyingThings3DSubsetDataset': FlyingThings3DSubsetDataset,
    'MiddleburyDataset': MiddleburyDataset,
    'ETH3DDataset': ETH3DDataset
}


def build_dataloader(data_cfg, batch_size, is_dist, workers, pin_memory, mode):
    all_dataset = []
    for data_info in data_cfg.DATA_INFOS:
        dataset = __all__[data_info.DATASET](
            data_info=data_info,
            data_cfg=data_cfg,
            mode=mode)
        all_dataset.append(dataset)
    dataset = torch.utils.data.ConcatDataset(all_dataset)

    shuffle = True if mode=='training' else False
    if is_dist:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    else:
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

    batch_uniform = data_cfg.DATA_TRANSFORM.get('BATCH_UNIFORM', False)
    batch_uniform = batch_uniform if mode=='training' else False
    random_type = data_cfg.DATA_TRANSFORM.get('RANDOM_TYPE', False)
    h_range = data_cfg.DATA_TRANSFORM.get('H_RANGE', False)
    w_range = data_cfg.DATA_TRANSFORM.get('W_RANGE', False)

    partial_custom_collate = partial(custom_collate, concat_dataset=dataset,
                                     batch_uniform=batch_uniform, random_type=random_type,
                                     h_range=h_range, w_range=w_range)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        collate_fn=partial_custom_collate,
        pin_memory=pin_memory,
        drop_last=False
    )
    return dataset, loader, sampler