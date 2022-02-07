from functools import partial

import numpy as np
from skimage import transform

from ...utils import box_utils, common_utils


class DataProcessor(object):
    """
    数据预处理类
    Args:
        processor_configs: DATA_CONFIG.DATA_PROCESSOR
        point_cloud_range:点云范围
        training:训练模式
    """
    def __init__(self, processor_configs, point_cloud_range, training):
        self.point_cloud_range = point_cloud_range # [0, -39.68, -3, 69.12, 39.68, 1]
        self.training = training
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []
        # 工厂模式，根据不同的配置，只需要增加相应的方法即可实现不同的调用
        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            # 在forward函数中调用
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        """
        移除超出point_cloud_range的点
        """
        # 偏函数是将所要承载的函数作为partial()函数的第一个参数，
        # 原函数的各个参数依次作为partial()函数后续的参数
        # 以便函数能用更少的参数进行调用
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        if data_dict.get('points', None) is not None:
            # mask为bool值，将x和y超过规定范围的点设置为0
            mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
            # 根据mask提取点
            data_dict['points'] = data_dict['points'][mask]

        # 当data_dict存在gt_boxes并且REMOVE_OUTSIDE_BOXES=True并且处于训练模式
        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            # mask为bool值，将box角点在范围内点个数大于最小阈值的设置为1
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        """将点云打乱"""
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0]) # 生成随机序列
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def transform_points_to_voxels(self, data_dict=None, config=None, voxel_generator=None):
        """
        将点云转换为voxel,调用spconv的VoxelGeneratorV2
        """
        if data_dict is None:
            try:
                from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            except:
                from spconv.utils import VoxelGenerator

            voxel_generator = VoxelGenerator(
                voxel_size=config.VOXEL_SIZE, # [0.16, 0.16, 4]
                point_cloud_range=self.point_cloud_range, # [0, -39.68, -3, 69.12, 39.68, 1]
                max_num_points=config.MAX_POINTS_PER_VOXEL, # 32
                max_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode] # 16000
            )
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE) # 网格数量
            self.grid_size = np.round(grid_size).astype(np.int64) 
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels, voxel_generator=voxel_generator)

        # 调用spconv的voxel_generator的generate方法生成体素
        points = data_dict['points']
        voxel_output = voxel_generator.generate(points)
        """
            voxels: (num_voxels, max_points_per_voxel, 3 + C)
            coordinates: (num_voxels, 3)
            num_points: (num_voxels)
        """
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = \
                voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
        else:
            voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        """
        采样点云，多了丢弃，少了补上
        """
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        # 如果采样点数 < 点云点数
        if num_points < len(points):
            # 计算点云深度
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1) # （N,）
            # 根据深度构造mask
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]  
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            # 如果采样点数 > 远点数量
            if num_points > len(far_idxs_choice):
                # 在近点中随机采样，因为近处稠密
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                # 如果远点不为0,则将采样的近点和远点拼接，如果为0,则直接返回采样的近点
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            # 如果采样点数 > 远点数量， 则直接随机采样
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            # 将点打乱
            np.random.shuffle(choice)
        # 如果采样点数 > 点云点数, 则随机采样点补全点云
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                # 随机采样缺少的点云索引
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                # 拼接索引
                choice = np.concatenate((choice, extra_choice), axis=0)
            # 将索引打乱
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def calculate_grid_size(self, data_dict=None, config=None):
        """
        计算网格范围
        """
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):
        """降采样深度图"""
        if data_dict is None:
            self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
            return partial(self.downsample_depth_map, config=config)
        # skimage中类似平均池化的操作，进行图像将采样
        data_dict['depth_maps'] = transform.downscale_local_mean(
            image=data_dict['depth_maps'],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor)
        )
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        # 在for循环中逐个流程处理，最终都放入data_dict中
        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
