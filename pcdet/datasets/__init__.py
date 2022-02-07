import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler

from pcdet.utils import common_utils

from .dataset import DatasetTemplate
from .kitti.kitti_dataset import KittiDataset
from .nuscenes.nuscenes_dataset import NuScenesDataset
from .waymo.waymo_dataset import WaymoDataset

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'KittiDataset': KittiDataset,
    'NuScenesDataset': NuScenesDataset,
    'WaymoDataset': WaymoDataset
}


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator() # 手动创建随机数生成器
            g.manual_seed(self.epoch) # 设置随机数种子
            indices = torch.randperm(len(self.dataset), generator=g).tolist() # 生成0-n的随机数排列
        else:
            indices = torch.arange(len(self.dataset)).tolist() # 如果不打乱则按顺序产生索引

        indices += indices[:(self.total_size - len(indices))] # 如果total_size比indices长则重复采样indices
        assert len(indices) == self.total_size # 确保indices的长度和total_size一样长

        indices = indices[self.rank:self.total_size:self.num_replicas] # 间隔采样
        assert len(indices) == self.num_samples # 确保当前采样长度和num_samples一样长

        return iter(indices) # 生成迭代器对象，可通过__next__方法逐个生成


def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0):
    """
    构建数据集并调用DataLoader进行加载
    Args:
        dataset_cfg: 数据集配置文件
        class_names: 类比名称
        batch_size: batch的大小
        dist: 是否并行训练
        root_path: 根目录
        workers: 线程数
        logger: 日志记录器
        training: 训练模式
        merge_all_iters_to_one_epoch: 是否将所有迭代次数合并到一个epoch
        total_epochs: 总epoch数
    Returns
        dataset: 数据集
        dataloader: 数据加载器
        sampler: 数据采样器
    """
    # 根据数据集名称，初始化数据集，即只执行__init__函数
    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info() # 获取rank和world_size
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False) # 初始化分布式采样器
    else:
        sampler = None
    # 初始化DataLoader，此时并没有进行数据采样和加载，只有在训练中才会按照batch size调用__getitem__加载数据
    # 在单卡训练中进行，通过DataLoader进行数据加载
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0
    )

    return dataset, dataloader, sampler
