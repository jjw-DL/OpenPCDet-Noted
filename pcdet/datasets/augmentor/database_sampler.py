import pickle

import numpy as np

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils


class DataBaseSampler(object):
    def __init__(self, root_path, sampler_cfg, class_names, logger=None):
        """
        真值采样增强类
        Args:
            root_path: 根目录
            sampler_cfg: 采样增强配置
            class_names: 类别名称
            logger: 日志
        """
        self.root_path = root_path
        self.class_names = class_names
        self.sampler_cfg = sampler_cfg
        self.logger = logger
        self.db_infos = {} # 数据库信息
        # 根据类别信息初始化空列表
        for class_name in class_names:
            self.db_infos[class_name] = []
        # sampler_cfg.DB_INFO_PATH: kitti_dbinfos_train.pkl
        for db_info_path in sampler_cfg.DB_INFO_PATH:
            db_info_path = self.root_path.resolve() / db_info_path
            # 按照类别读取信息
            """
            Car 00000
            {'name': 'Car', 'path': 'gt_database/000003_Car_0.bin', 'image_idx': '000003', 'gt_idx': 0, 
            'box3d_lidar': array([13.51070213, -0.98177999, -0.90948981,  4.15, 1.73, 1.57, -3.19079633]), 
            'num_points_in_gt': 677, 'difficulty': 0, 'bbox': array([614.24, 181.78, 727.31, 284.77], dtype=float32), 
            'score': -1.0}
            """
            with open(str(db_info_path), 'rb') as f:
                infos = pickle.load(f)
                [self.db_infos[cur_class].extend(infos[cur_class]) for cur_class in class_names]

        # 预处理，滤除点数少和识别困难的点云
        for func_name, val in sampler_cfg.PREPARE.items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val)

        self.sample_groups = {} # 采样分组信息
        self.sample_class_num = {} # 每类采样数量
        self.limit_whole_scene = sampler_cfg.get('LIMIT_WHOLE_SCENE', False)
        for x in sampler_cfg.SAMPLE_GROUPS: # ['Car:15','Pedestrian:15', 'Cyclist:15']
            class_name, sample_num = x.split(':')
            if class_name not in class_names:
                continue
            self.sample_class_num[class_name] = sample_num
            self.sample_groups[class_name] = {
                'sample_num': sample_num, # 采样个数
                'pointer': len(self.db_infos[class_name]), # 该类别总数
                'indices': np.arange(len(self.db_infos[class_name])) # 采样索引
            }

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def filter_by_difficulty(self, db_infos, removed_difficulty):
        """
        将difficulty=-1的infos过滤
        """
        new_db_infos = {}
        for key, dinfos in db_infos.items(): # 逐类别过滤
            pre_len = len(dinfos)
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
            # 记录过滤后的不同类别的infos长度
            if self.logger is not None:
                self.logger.info('Database filter by difficulty %s: %d => %d' % (key, pre_len, len(new_db_infos[key])))
        return new_db_infos

    def filter_by_min_points(self, db_infos, min_gt_points_list):
        """
        过滤点云数量较少的infos
        """
        for name_num in min_gt_points_list: # ['Car:5', 'Pedestrian:5', 'Cyclist:5']     
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            if min_num > 0 and name in db_infos.keys():
                filtered_infos = []
                for info in db_infos[name]: # 逐类别过滤
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)
                # 记录过滤后的不同类别的infos长度
                if self.logger is not None:
                    self.logger.info('Database filter by min points %s: %d => %d' %
                                     (name, len(db_infos[name]), len(filtered_infos)))
                db_infos[name] = filtered_infos

        return db_infos

    def sample_with_fixed_number(self, class_name, sample_group):
        """
        随机采样指定数量的infos
        Args:
            class_name: 类别名称
            sample_group: 采样组
        Returns:
            sampled_dict: 采样db_info的list
        """
        sample_num, pointer, indices = int(sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
        # 随机打乱索引
        if pointer >= len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0
        # 获取前sample_num个db_infos
        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]]
        # 采样个数
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices # 索引
        return sampled_dict

    @staticmethod
    def put_boxes_on_road_planes(gt_boxes, road_planes, calib):
        """
        Only validate in KITTIDataset
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            road_planes: [a, b, c, d]
            calib:

        Returns:
        """
        a, b, c, d = road_planes # 路面方程信息
        center_cam = calib.lidar_to_rect(gt_boxes[:, 0:3]) # 将box的中心点转换到相机坐标系下
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b # 计算cam距离路面的高度
        center_cam[:, 1] = cur_height_cam
        cur_lidar_height = calib.rect_to_lidar(center_cam)[:, 2] # 计算雷达距离地面的高度
        mv_height = gt_boxes[:, 2] - gt_boxes[:, 5] / 2 - cur_lidar_height # 将box的地面z坐标减去lidar高度
        gt_boxes[:, 2] -= mv_height  # lidar view  将box放到地面上
        return gt_boxes, mv_height

    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict):
        # 首先根据box mask过滤不符合要求的dict
        gt_boxes_mask = data_dict['gt_boxes_mask']
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask] # box
        gt_names = data_dict['gt_names'][gt_boxes_mask] # 类别
        points = data_dict['points'] # 点云
        # 如果USE_ROAD_PLANE，调用函数转换Z坐标
        if self.sampler_cfg.get('USE_ROAD_PLANE', False):
            sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
            )
            data_dict.pop('calib')
            data_dict.pop('road_plane')

        obj_points_list = []
        # 逐个采样box处理
        for idx, info in enumerate(total_valid_sampled_dict):
            file_path = self.root_path / info['path'] # data/kitti/gt_database/000003_Car_0.bin
            obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                [-1, self.sampler_cfg.NUM_POINT_FEATURES])

            obj_points[:, :3] += info['box3d_lidar'][:3] # 还原绝对坐标

            if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                # mv height
                obj_points[:, 2] -= mv_height[idx]

            obj_points_list.append(obj_points) # 将物体点云加入点云列表

        obj_points = np.concatenate(obj_points_list, axis=0) # 将点云拼接（M, N, 3）
        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict]) # 获取物体的类别

        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        ) # 将采样的box扩大，sampler_cfg.REMOVE_EXTRA_WIDTH即为dx，dy和dz的放大长度
        points = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes) # 将box内的点移除
        points = np.concatenate([obj_points, points], axis=0) # 将物体点云和放大后的box点云拼接，组成新的点云
        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0) # 将类别拼接
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0) # 将box拼接
        # 用新的box,类别和点云更新data_dict
        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict['points'] = points
        return data_dict

    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        gt_boxes = data_dict['gt_boxes'] # (N, 7 + C)
        gt_names = data_dict['gt_names'].astype(str) # (N,)
        existed_boxes = gt_boxes # 已经存在的boxes
        total_valid_sampled_dict = [] # 全部有效的采样字典
        # 逐类别进行采样
        for class_name, sample_group in self.sample_groups.items():
            if self.limit_whole_scene: # 限制整个点云中box的数量
                num_gt = np.sum(class_name == gt_names)
                sample_group['sample_num'] = str(int(self.sample_class_num[class_name]) - num_gt) # 计算要采样的数量
            if int(sample_group['sample_num']) > 0: # 如果采样box的数量大于0
                sampled_dict = self.sample_with_fixed_number(class_name, sample_group) # 在db_infos[calss_name]中随机采样sample_num个box信息

                sampled_boxes = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32) # 将采样box的box3d_lidar信息拼接

                if self.sampler_cfg.get('DATABASE_WITH_FAKELIDAR', False):
                    sampled_boxes = box_utils.boxes3d_kitti_fakelidar_to_lidar(sampled_boxes)

                iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], existed_boxes[:, 0:7]) # 计算采样box和已经存在的box的iou3d
                iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7]) # 计算采样box之间的iou3d
                iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0 # 将对角线设置为0
                iou1 = iou1 if iou1.shape[1] > 0 else iou2 # 如果已存在的box的个数 > 0
                valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0).nonzero()[0] # IoU为0(不存在碰撞)
                valid_sampled_dict = [sampled_dict[x] for x in valid_mask] # 选出有效的sampled_dict
                valid_sampled_boxes = sampled_boxes[valid_mask] # 选出有效的

                existed_boxes = np.concatenate((existed_boxes, valid_sampled_boxes), axis=0) # 将有效采样box和已经存在的box进行拼接
                total_valid_sampled_dict.extend(valid_sampled_dict) # 将有效的sampled_dict追加到总dict

        sampled_gt_boxes = existed_boxes[gt_boxes.shape[0]:, :] # 取出采样box
        if total_valid_sampled_dict.__len__() > 0:
            data_dict = self.add_sampled_boxes_to_scene(data_dict, sampled_gt_boxes, total_valid_sampled_dict) # 将采样box加入场景

        data_dict.pop('gt_boxes_mask')
        return data_dict
