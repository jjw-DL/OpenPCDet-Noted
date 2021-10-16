from collections import defaultdict # 当字典中的key不存在但被查找时，返回默认值，而不是keyError
from pathlib import Path

import numpy as np
import torch.utils.data as torch_data

from ..utils import common_utils
from .augmentor.data_augmentor import DataAugmentor
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder


class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg # 数据集配置文件
        self.training = training # 训练模式
        self.class_names = class_names # 类别
        self.logger = logger # 日志
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH) # 数据集根目录
        if self.dataset_cfg is None or class_names is None:
            return

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32) # 点云范围
        # 创建点云特征编码器类
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        # 创建数据增强器类
        self.data_augmentor = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger
        ) if self.training else None
        # 创建数据预处理器类
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range, training=self.training
        )

        self.grid_size = self.data_processor.grid_size # 网格数量 = 点云范围 / 体素大小
        self.voxel_size = self.data_processor.voxel_size # 体素大小
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

        if hasattr(self.data_processor, "depth_downsample_factor"):
            self.depth_downsample_factor = self.data_processor.depth_downsample_factor
        else:
            self.depth_downsample_factor = None

    
    @property
    def mode(self):
        """@property 可以让对象像访问属性一样区访问方法 self.mode"""
        return 'train' if self.training else 'test'

    def __getstate__(self):
        """Return state values to be pickled
        获取对象的属性（__init__中定义的属性,可以使用self.__dict__获取），返回去掉'logger'的属性dict
        """
        d = dict(self.__dict__) 
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d) # 根据字典d更新类的属性值

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        # @staticmethod不需要表示自身对象的self和自身类的cls参数，就和使用函数一样
        # @classmethod也不需要self参数，但第一个参数需要是表示自身类的cls参数
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:

        """

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        """
        合并所有的iters到一个epoch中
        """
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        # 类似c++中的虚函数，子类如果继承必须重写
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    def prepare_data(self, data_dict):
        """
        接受统一坐标系下的数据字典（points，box和class），进行数据筛选，数据预处理，包括数据增强，点云编码等
        Args:
            data_dict:
                points: optional, (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        # 训练模式下，对存在于class_name中的数据进行增强
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            # 返回一个bool数组，记录自定义数据集中ground_truth_name列表在不在我们需要检测的类别列表self.class_name里面
            # 比如kitti数据集中data_dict['gt_names']='car','person','cyclist'
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)
            # 数据增强 传入字典参数，**data_dict是将data_dict里面的key-value对都拿出来
            # 下面在原数据的基础上增加gt_boxes_mask，构造新的字典传入data_augmentor的forward函数
            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )

        # 筛选需要检测的gt_boxes
        if data_dict.get('gt_boxes', None) is not None:
            # 返回data_dict[gt_names]中存在于class_name的下标(np.array)
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            # 根据selected，选取需要的gt_boxes和gt_names
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            # 将当帧数据的gt_names中的类别名称对应到class_names的下标
            # 举个栗子，我们要检测的类别class_names = 'car','person'
            # 对于当前帧，类别gt_names = 'car', 'person', 'car', 'car'，当前帧出现了3辆车，一个人，获取索引后，gt_classes = 1, 2, 1, 1
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            # 将类别index信息放到每个gt_boxes的最后
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

            # 如果box2d不同，根据selected，选取需要的box2d
            if data_dict.get('gt_boxes2d', None) is not None:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][selected]

        # 使用点的哪些属性 比如x,y,z等
        if data_dict.get('points', None) is not None:
            data_dict = self.point_feature_encoder.forward(data_dict)

        # 对点云进行预处理，包括移除超出point_cloud_range的点、 打乱点的顺序以及将点云转换为voxel
        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        if self.training and len(data_dict['gt_boxes']) == 0:
            """
                如果处于训练模式，并且数据中含有gt_boxes
                首先，在数据长度范围内产生一个随机数
                然后调用__getitem__方法获取该索引的数据字典
            """
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        data_dict.pop('gt_names', None) # pop() 方法删除字典给定键 key 及对应的值，返回值为被删除的值

        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        """
        由于训练集中不同的点云的gt框个数不同，需要重写collate_batch函数，
        将不同item的boxes和labels等key放入list，返回batch_size的数据
        """
        # defaultdict创建一个带有默认返回值的字典，当key不存在时，返回默认值，list默认返回一个空
        data_dict = defaultdict(list)
        # 把batch里面的每个sample按照key-value合并
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        # 将合并后的key内的value进行拼接，先获取最大值，构造空矩阵，不足的部分补0
        # 因为pytorch要求输入数据维度一致
        for key, val in data_dict.items():
            try:
                # voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                # voxel_coords: optional (num_voxels, 3)
                # voxel_num_points: optional (num_voxels)
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        # 在每个坐标前面加上序号
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        """
                            ((0,0),(1,0))
                            在二维数组array第一维（此处便是行）前面填充0行，最后面填充0行；
                            在二维数组array第二维（此处便是列）前面填充1列，最后面填充0列
                            mode='constant'表示指定填充的参数
                            constant_values=i 表示第一维填充i
                        """
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0) # （B, N, 4)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val]) # 获取一个batch中所有帧中3D box最大的数量
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32) # 构造空的box3d矩阵（B, N, 7）
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k] #val[k]表示一个batch中的第k帧
                    ret[key] = batch_gt_boxes3d
                # gt_boxes2d同gt_boxes
                elif key in ['gt_boxes2d']: 
                    max_boxes = 0
                    max_boxes = max([len(x) for x in val])
                    batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32) # (B, N, 4)
                    for k in range(batch_size):
                        if val[k].size > 0:
                            batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_boxes2d
                elif key in ["images", "depth_maps"]:
                    # Get largest image size (H, W)
                    max_h = 0
                    max_w = 0
                    for image in val:
                        max_h = max(max_h, image.shape[0])
                        max_w = max(max_w, image.shape[1])

                    # Change size of images
                    images = []
                    for image in val:
                        pad_h = common_utils.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                        pad_w = common_utils.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                        pad_width = (pad_h, pad_w)
                        # Pad with nan, to be replaced later in the pipeline.
                        pad_value = np.nan

                        if key == "images":
                            pad_width = (pad_h, pad_w, (0, 0))
                        elif key == "depth_maps":
                            pad_width = (pad_h, pad_w)

                        image_pad = np.pad(image,
                                           pad_width=pad_width,
                                           mode='constant',
                                           constant_values=pad_value)

                        images.append(image_pad)
                    ret[key] = np.stack(images, axis=0) # (B, H, W, C)
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret
