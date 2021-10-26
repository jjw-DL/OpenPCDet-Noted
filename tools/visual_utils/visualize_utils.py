import mayavi.mlab as mlab
import numpy as np
import torch

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    将box角点沿Z轴旋转一定角度
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    # 将数据转换为tensor格式
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle) #（B,）
    sina = torch.sin(angle) #（B,）
    zeros = angle.new_zeros(points.shape[0]) #（B,）
    ones = angle.new_ones(points.shape[0]) #（B,）
    # 构造旋转矩阵
    # stack()，沿着一个新维度对输入张量序列进行连接，序列中所有的张量都应该为相同形状，
    # 会增加新的维度进行堆叠
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float() # （B, 3, 3）
    # 旋转box的角点
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    # 一般torch.cat()把torch.stack()得到tensor进行拼接
    # inputs:待连接的张量序列，可以是任意相同shape和Tensor类型的python序列
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1) # 这里是为了保证安全性和完整性，一般角点都是3维的
    # 转换为numpy格式并返回
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
        corners3d: (N, 8, 3)
    """
    # 首先转化numpy格式
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    # 定义8个角点box的模板
    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    # box的长宽高×template得到实际box的角点坐标
    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    # 将box旋转heading角
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    # 加中心点坐标
    corners3d += boxes3d[:, None, 0:3]
    # 返回box角点坐标
    return corners3d.numpy() if is_numpy else corners3d


def visualize_pts(pts, fig=None, bgcolor=(0, 0, 0), fgcolor=(1.0, 1.0, 1.0),
                  show_intensity=False, size=(600, 600), draw_origin=True):
    # 将点云转化为numpy格式
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()
    # 定义画布背景色，大小等信息
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=fgcolor, engine=None, size=size)

    # 调用points3d绘制点云
    if show_intensity:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    else:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    # 绘制原点和坐标轴
    if draw_origin:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
        mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), tube_radius=0.1)

    return fig


def draw_sphere_pts(pts, color=(0, 1, 0), fig=None, bgcolor=(0, 0, 0), scale_factor=0.2):
    """
    画球形点\n
    Args:
        pts: 待绘制点坐标\n
        color: 颜色\n
        fig: 画布\n
        bgcolor: 背景色\n
        scale_factor: 缩放比例\n
    Return:
        画布
    """
    # 1.将点转换为numpy格式
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()

    # 定义画布信息
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(600, 600))

    # 转换color数据格式
    if isinstance(color, np.ndarray) and color.shape[0] == 1:
        color = color[0]
        color = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
    
    if isinstance(color, np.ndarray):
        # 为所有点赋予相同颜色
        pts_color = np.zeros((pts.__len__(), 4), dtype=np.uint8)
        pts_color[:, 0:3] = color
        pts_color[:, 3] = 255
        # you can pass a fourth array s of the same shape as x, y, and z 
        # giving an associated scalar value for each point, or a function
        # f(x, y, z) returning the scalar value. This scalar value can be
        # used to modulate the color and the size of the points
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], np.arange(0, pts_color.__len__()), mode='sphere',
                          scale_factor=scale_factor, figure=fig)
        G.glyph.color_mode = 'color_by_scalar'
        G.glyph.scale_mode = 'scale_by_vector'
        G.module_manager.scalar_lut_manager.lut.table = pts_color
    else:
        mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='sphere', color=color,
                      colormap='gnuplot', scale_factor=scale_factor, figure=fig)
    # 绘制点和坐标轴
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
    mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), line_width=3, tube_radius=None, figure=fig)
    mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), line_width=3, tube_radius=None, figure=fig)
    mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), line_width=3, tube_radius=None, figure=fig)

    return fig


def draw_grid(x1, y1, x2, y2, fig, tube_radius=None, color=(0.5, 0.5, 0.5)):
    # plot3d要求逐个给出绘制点的x,y,z坐标
    mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    return fig


def draw_multi_grid_range(fig, grid_size=20, bv_range=(-60, -60, 60, 60)):
    # 计算网格点，并绘制网格线
    for x in range(bv_range[0], bv_range[2], grid_size):
        for y in range(bv_range[1], bv_range[3], grid_size):
            fig = draw_grid(x, y, x + grid_size, y + grid_size, fig)

    return fig


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_scores=None, ref_labels=None):
    """
    绘制点云
    Args:
        points:点云
        gt_boxes:真值box （N, 7）
        ref_boxes:预测box （M, 7）
        ref_scores:预测分数 (M,)
        ref_labels:预测类别 (M,)
    """
    # 1.判断数据类型，并将数据从tensor转化为numpy的array
    if not isinstance(points, np.ndarray):
        points = points.cpu().numpy()
    if ref_boxes is not None and not isinstance(ref_boxes, np.ndarray):
        ref_boxes = ref_boxes.cpu().numpy()
    if gt_boxes is not None and not isinstance(gt_boxes, np.ndarray):
        gt_boxes = gt_boxes.cpu().numpy()
    if ref_scores is not None and not isinstance(ref_scores, np.ndarray):
        ref_scores = ref_scores.cpu().numpy()
    if ref_labels is not None and not isinstance(ref_labels, np.ndarray):
        ref_labels = ref_labels.cpu().numpy()

    # 2.绘制点云
    fig = visualize_pts(points)
    # 3.绘制网格范围
    fig = draw_multi_grid_range(fig, bv_range=(0, -40, 80, 40))
    # 4.如果存在真值box，则绘制box
    if gt_boxes is not None:
        # 首先，将box从中心点+宽高形式转换为8个角点形式
        corners3d = boxes_to_corners_3d(gt_boxes)
        # 然后，绘制box
        fig = draw_corners3d(corners3d, fig=fig, color=(0, 0, 1), max_num=100)
    # 5.如果存在预测boxes，绘制预测box
    if ref_boxes is not None and len(ref_boxes) > 0:
        ref_corners3d = boxes_to_corners_3d(ref_boxes)
        if ref_labels is None:
            fig = draw_corners3d(ref_corners3d, fig=fig, color=(0, 1, 0), cls=ref_scores, max_num=100)
        else:
            # 按照类比逐类绘制，比如kitti数据是3类，k取0,1,2
            for k in range(ref_labels.min(), ref_labels.max() + 1):
                cur_color = tuple(box_colormap[k % len(box_colormap)]) # 根据k和box_colormap生成颜色
                mask = (ref_labels == k) # 获取mask
                # 绘制该类比的box
                fig = draw_corners3d(ref_corners3d[mask], fig=fig, color=cur_color, cls=ref_scores[mask], max_num=100)
    # 设定视角
    mlab.view(azimuth=-179, elevation=54.0, distance=104.0, roll=90.0)
    # 返回画布
    return fig


def draw_corners3d(corners3d, fig, color=(1, 1, 1), line_width=2, cls=None, tag='', max_num=500, tube_radius=None):
    """
    :param corners3d: (N, 8, 3)
    :param fig: 画布
    :param color: 颜色
    :param line_width: 线宽
    :param cls: 类别
    :param tag: 
    :param max_num: 最大数量
    :return:
    """
    import mayavi.mlab as mlab
    # 取绘制box的数量，并用最大值截断
    num = min(max_num, len(corners3d))
    # 逐个box进行绘制
    for n in range(num):
        b = corners3d[n]  # (8, 3)
        # 如果存在类别，在右下角写box的类别，
        # 在pcdet中定义box的前向为x轴，按照右手定则定y轴，朝上为z轴
        if cls is not None:
            if isinstance(cls, np.ndarray):
                mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%.2f' % cls[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)
            else:
                mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%s' % cls[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)
        # 每次绘制3条线，4次绘制12条线
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k, k + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

        # 绘制表示朝向的交叉线
        i, j = 0, 5
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)
        i, j = 1, 4
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)

    return fig
