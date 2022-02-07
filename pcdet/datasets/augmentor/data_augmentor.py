from functools import partial

import numpy as np

from ...utils import common_utils
from . import augmentor_utils, database_sampler


class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        """
        数据增强类
        Args:
            root_path: 根目录
            augmentor_configs: 增强器配置
            class_names: 类别名
            logger:日志
        """
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger
        # 数据增强器队列
        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                # 将被禁止的增强操作跳过
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            # 根据名称和配置，获取增强器
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, config=None):
        """
        ground truth 采样增强，调用database_sampler的DataBaseSampler处理（SECOND原创）
        """
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def random_world_flip(self, data_dict=None, config=None):
        """
        随机翻转
        """
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        # 获取gt_boxes和points
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        # 确定翻转轴
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            # 调用augmentor_utils中的函数翻转box和点云
            gt_boxes, points = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        """
        随机旋转
        """
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        # 获取旋转范围
        rot_range = config['WORLD_ROT_ANGLE'] # [-0.78539816, 0.78539816]
        # 如果旋转范围不是列表，则取正负
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        # 调用augmentor_utils中的函数旋转box和点云
        gt_boxes, points = augmentor_utils.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        """
        随机缩放
        """
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        # 调用augmentor_utils中的函数缩放box和点云
        gt_boxes, points = augmentor_utils.global_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE'] # [0.95, 1.05]
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_image_flip(self, data_dict=None, config=None):
        """"
        随机图片翻转
        """
        if data_dict is None:
            return partial(self.random_image_flip, config=config)
        # 获取图像，深度图，3Dbox和2Dbox以及abrading信息
        images = data_dict["images"]
        depth_maps = data_dict["depth_maps"]
        gt_boxes = data_dict['gt_boxes']
        gt_boxes2d = data_dict["gt_boxes2d"]
        calib = data_dict["calib"]
        # 遍历翻转轴列表
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['horizontal'] # np.fliplr只能水平翻转
            images, depth_maps, gt_boxes = getattr(augmentor_utils, 'random_image_flip_%s' % cur_axis)(
                images, depth_maps, gt_boxes, calib,
            )

        data_dict['images'] = images
        data_dict['depth_maps'] = depth_maps
        data_dict['gt_boxes'] = gt_boxes
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        # 遍历增强队列，逐个增强器做数据增强
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        # 将方位角限制在[-pi,pi]
        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        # 将标定信息弹出
        if 'calib' in data_dict:
            data_dict.pop('calib')
        # 将地面信息弹出
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        # 筛选mask选中的信息，最后将mask信息删除
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            if 'gt_boxes2d' in data_dict:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][gt_boxes_mask]

            data_dict.pop('gt_boxes_mask')
        return data_dict
