import copy
import numpy as np

from ...utils import common_utils


def random_flip_along_x(gt_boxes, points):
    """
    沿着x轴随机翻转
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    # 随机选择是否翻转
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1] # y坐标翻转
        gt_boxes[:, 6] = -gt_boxes[:, 6] # 方位角翻转，直接取负数，因为方位角定义为与x轴的夹角（这里按照顺时针的方向取角度）
        points[:, 1] = -points[:, 1] # 点云y坐标翻转

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8] # 如果有速度，y方向速度翻转

    return gt_boxes, points


def random_flip_along_y(gt_boxes, points):
    """
    沿着y轴随机翻转
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    # 随机旋转是否翻转
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0] # x坐标翻转
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi) # 方位角加pi后，取负数（这里按照顺时针的方向取角度）
        points[:, 0] = -points[:, 0]#  点云x坐标取反

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7] # 如果有速度，x方向速度取反


    return gt_boxes, points


def global_rotation(gt_boxes, points, rot_range):
    """
    对点云和box进行整体旋转
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    # 在均匀分布中随机产生旋转角度
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    # 沿z轴旋转noise_rotation弧度，这里之所以取第0个，是因为rotate_points_along_z对batch进行处理，而这里仅处理单个点云
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    # 同样对box的坐标进行旋转
    gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
    # 对box的方位角进行累加
    gt_boxes[:, 6] += noise_rotation
    # 对速度进行旋转，由于速度仅有x和y两个维度，所以补出第三维度，增加batch维度后进行旋转
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
            np.array([noise_rotation])
        )[0][:, 0:2]

    return gt_boxes, points


def global_scaling(gt_boxes, points, scale_range):
    """
    随机缩放
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    # 如果缩放的尺度过小，则直接返回原来的box和点云
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    # 在缩放范围内随机产生缩放尺度
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    # 将点云和box同时乘以缩放尺度
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    return gt_boxes, points


def random_image_flip_horizontal(image, depth_map, gt_boxes, calib):
    """
    Performs random horizontal flip augmentation
    Args:
        image: (H_image, W_image, 3), Image
        depth_map: (H_depth, W_depth), Depth map
        gt_boxes: (N, 7), 3D box labels in LiDAR coordinates [x, y, z, w, l, h, ry]
        calib: calibration.Calibration, Calibration object
    Returns:
        aug_image: (H_image, W_image, 3), Augmented image
        aug_depth_map: (H_depth, W_depth), Augmented depth map
        aug_gt_boxes: (N, 7), Augmented 3D box labels in LiDAR coordinates [x, y, z, w, l, h, ry]
    """
    # Randomly augment with 50% chance
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])

    if enable:
        # Flip images
        aug_image = np.fliplr(image)
        aug_depth_map = np.fliplr(depth_map)

        # Flip 3D gt_boxes by flipping the centroids in image space
        aug_gt_boxes = copy.copy(gt_boxes)
        locations = aug_gt_boxes[:, :3] # 取出前3维中心点
        img_pts, img_depth = calib.lidar_to_img(locations) # 找到box在图片的中心点
        W = image.shape[1] # 获取图片宽度
        img_pts[:, 0] = W - img_pts[:, 0] # 计算中心点翻转后的坐标
        pts_rect = calib.img_to_rect(u=img_pts[:, 0], v=img_pts[:, 1], depth_rect=img_depth)
        pts_lidar = calib.rect_to_lidar(pts_rect) # 将中心点坐标转换回点云坐标系
        aug_gt_boxes[:, :3] = pts_lidar # 点云坐标
        aug_gt_boxes[:, 6] = -1 * aug_gt_boxes[:, 6] # box的方位角取反

    else:
        aug_image = image
        aug_depth_map = depth_map
        aug_gt_boxes = gt_boxes

    return aug_image, aug_depth_map, aug_gt_boxes