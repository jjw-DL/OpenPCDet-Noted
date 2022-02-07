import copy
import pickle

import numpy as np
from skimage import io

from . import kitti_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..dataset import DatasetTemplate

# 定义kitti数据集的类
class KittiDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        # 初始化类，将参数赋值给类的属性
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        # 传递参数是 训练集train 还是验证集val
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        # root_path的路径是../data/kitti/
        # kitti数据集一共三个文件夹“training”和“testing”、“ImageSets”
        # 如果是训练集train，将文件的路径指为训练集training ，否则为测试集testing
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        # /data/kitti/ImageSets/下面一共三个文件：test.txt , train.txt ,val.txt
        # 选择其中的一个文件
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        # 得到.txt文件下的序列号，组成列表sample_id_list
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
        # 创建用于存放kitti信息的空列表
        self.kitti_infos = []
        # 调用函数，加载kitti数据，mode的值为：train 或者  test
        self.include_kitti_data(self.mode)

    def include_kitti_data(self, mode):
        if self.logger is not None:
            # 如果日志信息存在，则加入'Loading KITTI dataset'的信息
            self.logger.info('Loading KITTI dataset')
        # 创建新列表，用于存放信息
        kitti_infos = []

        '''   
        INFO_PATH: {
        'train': [kitti_infos_train.pkl],
        'test': [kitti_infos_val.pkl],}
        '''
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            # root_path的路径是/data/kitti/
            info_path = self.root_path / info_path
            # 则 info_path：/data/kitti/kitti_infos_train.pkl之类的文件
            if not info_path.exists():
                # 如果该文件不存在，跳出，继续下一个文件
                continue
            # 打开该文件
            with open(info_path, 'rb') as f:
                # pickle.load(f) 将该文件中的数据解析为一个Python对象infos，
                # 并将该内容添加到kitti_infos列表中
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        self.kitti_infos.extend(kitti_infos)

        if self.logger is not None:
            self.logger.info('Total samples for KITTI dataset: %d' % (len(kitti_infos)))

    def set_split(self, split):
        """ get tag list according to the split

        Args:
            split(string): train or test

        Returns:
            list: list of tag

        """
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        # root_path的路径是/data/kitti/ 
        # 则root_split_path=/data/kitti/training
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        # /data/kitti/ImageSets/下面一共三个文件：test.txt , train.txt ,val.txt
        # 选择其中的一个文件
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        # 将文件中的tag构造为列表，方便处理
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, idx):
        """Loads point cloud for a sample
        Args:
            index (int): Index of the point cloud file to get.
        Returns:
            np.array(N, 4): point cloud.
        """
        # /data/kitti/training/velodyne/xxxxxx.bin
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_image(self, idx):
        """
        Loads image for a sample
        Args:
            idx: int, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        # /data/kitti/training/image_2/xxxxxx.png
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        image = io.imread(img_file)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_image_shape(self, idx):
        # /data/kitti/training/image_2/xxxxxx.png
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()  #如果该图片文件不存在，直接报错
        # 该函数的返回值是：array([375, 1242], dtype=int32)
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        # /data/kitti/training/label_2/xxxxxx.txt
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists() #如果不存在，直接报错
        # 调用get_objects_from_label函数，首先读取该文件的所有行赋值为 lines
        # 在对lines中的每一个line（一个object的参数）作为object3d类的参数 进行遍历，
        # 最后返回：objects[]列表 ,里面是当前文件里所有物体的属性值，如：type、x,y,等
        return object3d_kitti.get_objects_from_label(label_file)

    def get_depth_map(self, idx):
        """
        Loads depth map for a sample
        Args:
            idx: str, Sample index
        Returns:
            depth: (H, W), Depth map
        """
        # /data/kitti/training/depth_2/xxxxxx.txt
        depth_file = self.root_split_path / 'depth_2' / ('%s.png' % idx)
        assert depth_file.exists()
        depth = io.imread(depth_file)
        depth = depth.astype(np.float32)
        depth /= 256.0
        return depth

    def get_calib(self, idx):
        # /data/kitti/training/calib/xxxxxx.txt
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def get_road_plane(self, idx):
        # /data/kitti/training/planes/xxxxxx.txt
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """ 
        Args:
            pts_rect: 在rect系下的点云
            img_shape: 图像的尺寸
            calib: 标定信息

        Returns:
            true, if the point in the fov

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        # 判断投影点是否在图像范围内
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        # 深度 > 0, 才可以判断在fov视角
        # pts_valid_flag=array([ True,   True,  True, False,   True, True,.....])之类的，一共有M个 
        # 用于判断该点云能否有效 （是否用于训练）
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures
        # 线程函数
        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            # 定义info空字典
            info = {}
            # 点云信息：点云特征维度和索引
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            # 添加点云信息
            info['point_cloud'] = pc_info
            # 图像信息：索引和图像高宽
            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            # 添加图像信息
            info['image'] = image_info

            # 根据索引获取Calibration对象
            calib = self.get_calib(sample_idx)
            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            # 标定信息：P2、R0_rect和T_V_C
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}
            # 添加标定信息
            info['calib'] = calib_info

            if has_label:
                # 根据索引读取label，构造object列表
                obj_list = self.get_label(sample_idx)
                annotations = {}
                # 根据属性将所有obj_list的属性添加进annotations
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                # 计算有效物体的个数，如10个，object除去“DontCare”4个，还剩num_objects6个
                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                # 总物体的个数 10个
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                # 由此可以得到 index=[0,1,2,3,4,5,-1,-1,-1,-1]
                annotations['index'] = np.array(index, dtype=np.int32)

                # 假设有效物体的个数是N
                # 取有效物体的 location（N,3）、dimensions（N,3）、rotation_y（N,1）信息
                # kitti中'DontCare'一定放在最后,所以可以这样取值
                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                # 通过计算得到在lidar坐标系下的坐标，loc_lidar:（N,3）
                loc_lidar = calib.rect_to_lidar(loc)
                # 分别取 dims中的第一列、第二列、第三列：l,h,w（N,1）
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3] 
                loc_lidar[:, 2] += h[:, 0] / 2 # 将物体的坐标原点由物体底部中心移到物体中心
                # (N, 7) [x, y, z, dx, dy, dz, heading] 
                # np.newaxis在列上增加一维，因为rots是(N,)
                # -(np.pi / 2 + rots[..., np.newaxis]) 应为在kitti中，camera坐标系下定义物体朝向与camera的x轴夹角顺时针为正，逆时针为负
                # 在pcdet中，lidar坐标系下定义物体朝向与lidar的x轴夹角逆时针为正，顺时针为负，所以二者本身就正负相反
                # pi / 2是坐标系x轴相差的角度(如图所示)
                # camera:         lidar:
                # Y                    X
                # |                    |
                # |____X         Y_____|       
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar
                # 添加注释信息
                info['annos'] = annotations

                if count_inside_pts:
                    # 根据索引获取点云
                    points = self.get_lidar(sample_idx)
                    # 根据索引获取Calibration对象
                    calib = self.get_calib(sample_idx)
                    # 将lidar坐标系的点变换到rect坐标系
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])
                    # 返回true or false list判断点云是否在fov下，判断该点云能否有效 （是否用于训练）
                    fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    # 提取有效点
                    pts_fov = points[fov_flag]
                    # gt_boxes_lidar是(N,7)  [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
                    # 返回值corners_lidar为（N,8,3）
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    # num_gt是这一帧图像里物体的总个数，假设为10，
                    # 则num_points_in_gt=array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=int32)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    # num_objects是有效物体的个数，为N，假设为N=6
                    for k in range(num_objects):
                        # in_hull函数是判断点云是否在bbox中，(是否在物体的2D检测框中)
                        # 如果是，返回flag
                        # 运用到了“三角剖分”的概念和方法
                        # 输入是当前帧FOV视角点云和第K个box信息
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        # 计算框内包含的点云
                        num_points_in_gt[k] = flag.sum()
                    # 添加框内点云数量信息
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        # 创建线程池，多线程异步处理，增加处理速度
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        # infos是一个列表，每一个元素代表了一帧的信息（字典）
        return list(infos)

    # 用trainfile的groundtruth产生groundtruth_database，
    # 只保存训练数据中的gt_box及其包围的点的信息，用于数据增强
    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch
        # 如果是“train”，创建的路径是  /data/kitti/gt_database
        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        # 在/data/kitti/下创建保存kitti_dbinfos_train的文件
        db_info_save_path = Path(self.root_path) / ('kitti_dbinfos_%s.pkl' % split)
        # parents=True，可以同时创建多级目录
        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        # 传入的参数info_path是一个.pkl文件，ROOT_DIR/data/kitti/kitti_infos_train.pkl
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        # 读取infos里的每个info的信息，一个info是一帧的数据
        for k in range(len(infos)):
            # 输出的是 第几个样本 如7/780
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            # 取当前帧的信息 info
            info = infos[k]
            # 取里面的样本序列，其实就是data/kitti/ImageSets/train.txt里面的数字序列，
            # 如000000，000003,000007....
            sample_idx = info['point_cloud']['lidar_idx']
            # 读取该bin文件类型，并将点云数据以numpy的格式输出
            # points是一个数组（M,4）
            points = self.get_lidar(sample_idx)
            # 读取注释信息
            annos = info['annos']
            # name的数据是['car','car','pedestrian'...'dontcare'...]表示当前帧里面的所有物体objects
            names = annos['name']
            # difficulty：[0,1,2,-1,0,0,-1,1,...,]里面具体物体的难度，长度为总物体的个数
            difficulty = annos['difficulty']
            # bbox是一个数组，表示物体2D边框的个数，
            # 假设有效物体为N,dontcare个数为n,则bbox:(N+n,4）
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            # num_obj是有效物体的个数，为N
            num_obj = gt_boxes.shape[0]
            # 返回每个box中的点云索引[0 0 0 1 0 1 1...]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                # 创建文件名，并设置保存路径，最后文件如：000007_Cyclist_3.bin
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                # /data/kitti/gt_database/000007_Cyclist_3.bin
                filepath = database_save_path / filename
                # point_indices[i] > 0得到的是一个[T,F,T,T,F...]之类的真假索引，共有M个
                # 再从points中取出相应为true的点云数据，放在gt_points中
                gt_points = points[point_indices[i] > 0]

                # 将第i个box内点转化为局部坐标
                gt_points[:, :3] -= gt_boxes[i, :3]
                # 把gt_points的信息写入文件里
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    # 获取文件相对路径
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    # 根据当前物体的信息组成info
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    # 把db_info信息添加到 all_db_infos字典里面
                    if names[i] in all_db_infos:
                        # 如果存在该类别则追加
                        all_db_infos[names[i]].append(db_info)
                    else:
                        # 如果不存在该类别则新增
                        all_db_infos[names[i]] = [db_info]
        # 输出数据集中不同类别物体的个数
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))
        # 把所有的all_db_infos写入到文件里面
        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict: 
                为字典包含batch的calib和image_shape等信息，通过frame_id索引
            pred_dicts: list of pred_dicts 预测列表包含:
                pred_boxes: (N, 7), Tensor 预测的框，包含七个信息
                pred_scores: (N), Tensor   预测得分
                pred_labels: (N), Tensor   预测的类比
            class_names:
            output_path:

        Returns:

        """
        # 获取预测后的模板字典pred_dict，全部定义为全零的向量
        # kitti格式增加了score和boxes_lidar信息
        # 参数num_samples表示这一帧里面的物体个数
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            """
            接收模型预测的在统一坐标系下表示的3D检测框，并转回自己所需格式,生成一帧的预测字典
            Args:
                batch_index:batch的索引id
                box_dict:预测的结果，字典包含pred_scores、pred_boxes、pred_labels等信息
            """
            #pred_scores: (N), Tensor      预测得分，N是这一帧预测物体的个数
            #pred_boxes: (N, 7), Tensor    预测的框，包含七个信息
            #pred_labels: (N), Tensor      预测的标签
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            # 定义一个帧的空字典，用来存放来自预测的信息
            pred_dict = get_template_prediction(pred_scores.shape[0])
            # 如果没有物体，则返回空字典
            if pred_scores.shape[0] == 0:
                return pred_dict

            # 获取该帧的标定和图像尺寸信息
            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            # 将预测的box3D转化从lidar系转化到camera系
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            # 将camera系下的box3D信息转化为box2D信息
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )
            # 向刚刚创建的全零字典中填充预测信息，类别名，角度等信息（kitti格式）
            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            # 返回预测字典
            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            # 获取id帧号
            frame_id = batch_dict['frame_id'][index]
            # 获取单帧的预测结果
            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                # 定义输出结果的文件路径比如： data/kitti/output/000007.txt
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    # 从单帧预测dict中，提取bbox、loc和dims等信息
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl
                    # 将预测信息输出至终端同时写入文件
                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos # 返回所有预测信息

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.kitti_infos[0].keys():
            # 如果'annos'没在kitti信息里面，直接返回空字典，如果has_label=True则会构建annos
            return None, {}

        from .kitti_object_eval_python import eval as kitti_eval

        # 复制一下参数det_annos
        # copy.deepcopy()在元组和列表的嵌套上的效果是一样的，都是进行了深拷贝（递归的）
        eval_det_annos = copy.deepcopy(det_annos)
        # 一个info 表示一帧数据的信息，则下面是把所有数据的annos属性取出来，进行copy
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        # 根据目标检测的真值和预测值，计算四个检测指标 分别为 bbox、bev、3d和aos
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs
        #等于返回训练帧的总个数，等于图片的总个数，帧的总个数
        return len(self.kitti_infos)

    # 将点云与3D标注框均转至前述统一坐标定义下，送入数据基类提供的self.prepare_data()
    def __getitem__(self, index):
        """
        从pkl文件中获取相应index的info，然后根据info['point_cloud']['lidar_idx']确定帧号，进行数据读取和其他info字段的读取
        初步读取的data_dict,要传入prepare_data（dataset.py父类中定义）进行统一处理，然后即可返回
        """
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        # 取出第index帧的信息
        info = copy.deepcopy(self.kitti_infos[index])

        # 获取采样的序列号，在train.txt文件里的数据序列号
        sample_idx = info['point_cloud']['lidar_idx']
        # 获取该序列号相应的 图像宽高
        img_shape = info['image']['image_shape']
        # 获取该序列号相应的相机参数，如P2,R0,V2C
        calib = self.get_calib(sample_idx)
        # 获取item列表
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        # 定义输入数据的字典包含帧id和标定信息
        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            # 获取该帧信息中的 annos
            annos = info['annos']
            # 下面函数的作用是 在info中剔除包含'DontCare'的数据信息
            # 不但从name中剔除，余下的location、dimensions等信息也都不考虑在内
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            # 得到有效物体object(N个)的位置、大小和角度信息（N,3）,(N,3),(N)
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']

            # 构造camera系下的label（N,7），再转换到lidar系下
            # boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
            # boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            # 将新的键值对 添加到输入的字典中去，此时输入中有四个键值对了
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

            # 如果get_item_list中有gt_boxes2d，则将bbox加入到input_dict中
            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos["bbox"]
            
            #如果有路面信息，则加入进去
            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        # 加入点云，如果要求FOV视角，则对点云进行裁剪后加入input_dict
        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            if self.dataset_cfg.FOV_POINTS_ONLY:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]
            input_dict['points'] = points

        # 加入图片
        if "images" in get_item_list:
            input_dict['images'] = self.get_image(sample_idx)

        # 加入深度图
        if "depth_maps" in get_item_list:
            input_dict['depth_maps'] = self.get_depth_map(sample_idx)

        # 加入标定信息
        if "calib_matricies" in get_item_list:
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)

        # 将输入数据送入prepare_data进一步处理，形成训练数据
        data_dict = self.prepare_data(data_dict=input_dict)

        # 加入图片宽高信息
        data_dict['image_shape'] = img_shape
        return data_dict


def create_kitti_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    """
    生成.pkl文件（对train/test/val均生成相应文件），提前读取点云格式、image格式、calib矩阵以及label
    """
    dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'
    # 定义文件的路径和名称
    train_filename = save_path / ('kitti_infos_%s.pkl' % train_split) # /data/kitti/kitti_infos_train.pkl
    val_filename = save_path / ('kitti_infos_%s.pkl' % val_split) # /data/kitti/kitti_infos_val.pkl
    trainval_filename = save_path / 'kitti_infos_trainval.pkl'
    test_filename = save_path / 'kitti_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split) 
    # 执行完上一步，得到train相关的保存文件，以及sample_id_list的值为train.txt文件下的数字
    # 下面是得到train.txt中序列相关的所有点云数据的信息，并且进行保存
    kitti_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    print('Kitti info train file is saved to %s' % train_filename)

    # 对验证集的数据进行信息统计并保存
    dataset.set_split(val_split)
    kitti_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    print('Kitti info val file is saved to %s' % val_filename)

    # 将训练集和验证集的信息合并写到一个文件里
    with open(trainval_filename, 'wb') as f:
        pickle.dump(kitti_infos_train + kitti_infos_val, f)
    print('Kitti info trainval file is saved to %s' % trainval_filename)

    # 对测试集的数据进行信息统计并保存
    dataset.set_split('test')
    kitti_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(kitti_infos_test, f)
    print('Kitti info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    # 用trainfile产生groundtruth_database
    # 只保存训练数据中的gt_box及其包围的点的信息，用于数据增强
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_kitti_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve() # OpenPCDet根目录
        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'kitti', # OpenPCDet/data/kitti
            save_path=ROOT_DIR / 'data' / 'kitti' # OpenPCDet/data/kitti
        )