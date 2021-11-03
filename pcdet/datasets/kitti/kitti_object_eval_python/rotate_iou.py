#####################
# Based on https://github.com/hongzhenwang/RRPN-revise
# Licensed under The MIT License
# Author: yanyan, scrin@foxmail.com
#####################
import math

import numba
import numpy as np
from numba import cuda


@numba.jit(nopython=True)
def div_up(m, n):
    # 向上取整
    return m // n + (m % n > 0)

@cuda.jit('(float32[:], float32[:], float32[:])', device=True, inline=True)
def trangle_area(a, b, c):
    """
    计算三角形面积（叉乘法）
       /a
     c/___b    
    """
    return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) *(b[0] - c[0])) / 2.0


@cuda.jit('(float32[:], int32)', device=True, inline=True)
def area(int_pts, num_of_inter):
    area_val = 0.0
    # 逐多边形点遍历，求解面积
    for i in range(num_of_inter - 2):
        area_val += abs(
            trangle_area(int_pts[:2], int_pts[2 * i + 2:2 * i + 4],
                         int_pts[2 * i + 4:2 * i + 6]))
    return area_val


@cuda.jit('(float32[:], int32)', device=True, inline=True)
def sort_vertex_in_convex_polygon(int_pts, num_of_inter):
    if num_of_inter > 0:
        center = cuda.local.array((2, ), dtype=numba.float32)
        center[:] = 0.0
        # 1.计算多边形的平均中心点
        for i in range(num_of_inter):
            center[0] += int_pts[2 * i]
            center[1] += int_pts[2 * i + 1]
        center[0] /= num_of_inter
        center[1] /= num_of_inter
        # 2.对多边形的点排序
        v = cuda.local.array((2, ), dtype=numba.float32)
        vs = cuda.local.array((16, ), dtype=numba.float32)
        for i in range(num_of_inter):
            v[0] = int_pts[2 * i] - center[0]
            v[1] = int_pts[2 * i + 1] - center[1]
            d = math.sqrt(v[0] * v[0] + v[1] * v[1])
            v[0] = v[0] / d
            v[1] = v[1] / d
            if v[1] < 0:
                v[0] = -2 - v[0]
            vs[i] = v[0]
        j = 0
        temp = 0
        for i in range(1, num_of_inter):
            if vs[i - 1] > vs[i]:
                temp = vs[i]
                tx = int_pts[2 * i]
                ty = int_pts[2 * i + 1]
                j = i
                while j > 0 and vs[j - 1] > temp:
                    vs[j] = vs[j - 1]
                    int_pts[j * 2] = int_pts[j * 2 - 2]
                    int_pts[j * 2 + 1] = int_pts[j * 2 - 1]
                    j -= 1

                vs[j] = temp
                int_pts[j * 2] = tx
                int_pts[j * 2 + 1] = ty


@cuda.jit('(float32[:], float32[:], int32, int32, float32[:])',device=True,inline=True)
def line_segment_intersection(pts1, pts2, i, j, temp_pts):
    """
    求两条直线的交点
    Args：
        pts1:（8,）四个角点坐标
        pts2: (8,) 四个角点坐标
        i:box1的索引
        j:box2的索引
    Returns:
        temp_pts:返回交点信息
    """
    A = cuda.local.array((2, ), dtype=numba.float32)
    B = cuda.local.array((2, ), dtype=numba.float32)
    C = cuda.local.array((2, ), dtype=numba.float32)
    D = cuda.local.array((2, ), dtype=numba.float32)
    # 在box1中取出一条边
    A[0] = pts1[2 * i]
    A[1] = pts1[2 * i + 1]

    B[0] = pts1[2 * ((i + 1) % 4)]
    B[1] = pts1[2 * ((i + 1) % 4) + 1]

    # 在box2中取出一条边
    C[0] = pts2[2 * j]
    C[1] = pts2[2 * j + 1]

    D[0] = pts2[2 * ((j + 1) % 4)]
    D[1] = pts2[2 * ((j + 1) % 4) + 1]
    
    # 判断两条直线是否有交点 
    BA0 = B[0] - A[0] #（x2 - x1）
    BA1 = B[1] - A[1] #（y2 - y1）
    DA0 = D[0] - A[0] #（x4 - x1）
    CA0 = C[0] - A[0] #（x3 - x1）
    DA1 = D[1] - A[1] #（y4 - y1）
    CA1 = C[1] - A[1] #（y3 - y1）
    acd = DA1 * CA0 > CA1 * DA0
    bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
    # 如果存在交点则求交点
    if acd != bcd:
        abc = CA1 * BA0 > BA1 * CA0
        abd = DA1 * BA0 > BA1 * DA0
        if abc != abd:
            DC0 = D[0] - C[0]
            DC1 = D[1] - C[1]
            ABBA = A[0] * B[1] - B[0] * A[1]
            CDDC = C[0] * D[1] - D[0] * C[1]
            DH = BA1 * DC0 - BA0 * DC1
            Dx = ABBA * DC0 - BA0 * CDDC
            Dy = ABBA * DC1 - BA1 * CDDC
            temp_pts[0] = Dx / DH
            temp_pts[1] = Dy / DH
            return True
    return False


@cuda.jit(
    '(float32[:], float32[:], int32, int32, float32[:])',
    device=True,
    inline=True)
def line_segment_intersection_v1(pts1, pts2, i, j, temp_pts):
    a = cuda.local.array((2, ), dtype=numba.float32)
    b = cuda.local.array((2, ), dtype=numba.float32)
    c = cuda.local.array((2, ), dtype=numba.float32)
    d = cuda.local.array((2, ), dtype=numba.float32)

    a[0] = pts1[2 * i]
    a[1] = pts1[2 * i + 1]

    b[0] = pts1[2 * ((i + 1) % 4)]
    b[1] = pts1[2 * ((i + 1) % 4) + 1]

    c[0] = pts2[2 * j]
    c[1] = pts2[2 * j + 1]

    d[0] = pts2[2 * ((j + 1) % 4)]
    d[1] = pts2[2 * ((j + 1) % 4) + 1]

    area_abc = trangle_area(a, b, c)
    area_abd = trangle_area(a, b, d)

    if area_abc * area_abd >= 0:
        return False

    area_cda = trangle_area(c, d, a)
    area_cdb = area_cda + area_abc - area_abd

    if area_cda * area_cdb >= 0:
        return False
    t = area_cda / (area_abd - area_abc)

    dx = t * (b[0] - a[0])
    dy = t * (b[1] - a[1])
    temp_pts[0] = a[0] + dx
    temp_pts[1] = a[1] + dy
    return True


@cuda.jit('(float32, float32, float32[:])', device=True, inline=True)
def point_in_quadrilateral(pt_x, pt_y, corners):
    """
    判断一个box的角点是否在另一个box内
    a——————b
    |  .p  |
    |      |
    d——————c
    """
    ab0 = corners[2] - corners[0]
    ab1 = corners[3] - corners[1]

    ad0 = corners[6] - corners[0]
    ad1 = corners[7] - corners[1]

    ap0 = pt_x - corners[0]
    ap1 = pt_y - corners[1]

    abab = ab0 * ab0 + ab1 * ab1
    abap = ab0 * ap0 + ab1 * ap1
    adad = ad0 * ad0 + ad1 * ad1
    adap = ad0 * ap0 + ad1 * ap1
    # 只有以下内容全部为true，才证明该点在box内
    return abab >= abap and abap >= 0 and adad >= adap and adap >= 0


@cuda.jit('(float32[:], float32[:], float32[:])', device=True, inline=True)
def quadrilateral_intersection(pts1, pts2, int_pts):
    """
    计算两个box交集的角点坐标
    Args:
        pts1:(8,)
        pts2:(8,)
    Returns:
        int_pts:(16,)
    """
    num_of_inter = 0
    # 计算交集的角点
    for i in range(4):
        # 分别检查四个角点在对方box内
        if point_in_quadrilateral(pts1[2 * i], pts1[2 * i + 1], pts2):
            int_pts[num_of_inter * 2] = pts1[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1]
            num_of_inter += 1
        if point_in_quadrilateral(pts2[2 * i], pts2[2 * i + 1], pts1):
            int_pts[num_of_inter * 2] = pts2[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1]
            num_of_inter += 1
    temp_pts = cuda.local.array((2, ), dtype=numba.float32)
    
    for i in range(4):
        for j in range(4):
            # 求交点坐标
            has_pts = line_segment_intersection(pts1, pts2, i, j, temp_pts)
            if has_pts:
                int_pts[num_of_inter * 2] = temp_pts[0]
                int_pts[num_of_inter * 2 + 1] = temp_pts[1]
                num_of_inter += 1

    return num_of_inter


@cuda.jit('(float32[:], float32[:])', device=True, inline=True)
def rbbox_to_corners(corners, rbbox):
    # generate clockwise corners and rotate it clockwise
    # 将中心点加宽高的形式和角度的形式转换为角点形式(x1,y1,x2,y2,x3,y3,x4,y4)
    # 1.初始化
    angle = rbbox[4]
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    center_x = rbbox[0]
    center_y = rbbox[1]
    x_d = rbbox[2]
    y_d = rbbox[3]
    corners_x = cuda.local.array((4, ), dtype=numba.float32)
    corners_y = cuda.local.array((4, ), dtype=numba.float32)
    # 2.计算在坐标原点的box模板
    corners_x[0] = -x_d / 2
    corners_x[1] = -x_d / 2
    corners_x[2] = x_d / 2
    corners_x[3] = x_d / 2
    corners_y[0] = -y_d / 2
    corners_y[1] = y_d / 2
    corners_y[2] = y_d / 2
    corners_y[3] = -y_d / 2
    # 3.然后旋转和平移，计算4个角点
    for i in range(4):
        corners[2 *
                i] = a_cos * corners_x[i] + a_sin * corners_y[i] + center_x
        corners[2 * i
                + 1] = -a_sin * corners_x[i] + a_cos * corners_y[i] + center_y


@cuda.jit('(float32[:], float32[:])', device=True, inline=True)
def inter(rbbox1, rbbox2):
    # 0.初始化
    # 在cuda上开辟局部内存，存储角点信息
    corners1 = cuda.local.array((8, ), dtype=numba.float32)
    corners2 = cuda.local.array((8, ), dtype=numba.float32)
    # 由于交集部分不规则，但是最多8个点，共16个坐标
    intersection_corners = cuda.local.array((16, ), dtype=numba.float32)
    # 1.求两个box的四个角点坐标，先平移，然后旋转
    rbbox_to_corners(corners1, rbbox1)
    rbbox_to_corners(corners2, rbbox2)
    # 2.求box的所有交点
    num_intersection = quadrilateral_intersection(corners1, corners2,
                                                  intersection_corners)
    # 3.根据对交集多边形中心点对其角点进行排序，使其满足逆时针方向
    sort_vertex_in_convex_polygon(intersection_corners, num_intersection)
    # print(intersection_corners.reshape([-1, 2])[:num_intersection])
    # 4.根据叉乘几何意义，求取交集多边形面积（由于夹角逐渐增大，与第一个点构成的三角形面积一定是在同侧的符号相同）
    return area(intersection_corners, num_intersection)


@cuda.jit('(float32[:], float32[:], int32)', device=True, inline=True)
def devRotateIoUEval(rbox1, rbox2, criterion=-1):
    # 1.计算两个box的面积
    area1 = rbox1[2] * rbox1[3]
    area2 = rbox2[2] * rbox2[3]
    # 2.求两个box的交集
    area_inter = inter(rbox1, rbox2)
    # 3.求交并比
    if criterion == -1:
        return area_inter / (area1 + area2 - area_inter)
    elif criterion == 0:
        return area_inter / area1
    elif criterion == 1:
        return area_inter / area2
    else:
        return area_inter

@cuda.jit('(int64, int64, float32[:], float32[:], float32[:], int32)', fastmath=False)
def rotate_iou_kernel_eval(N, K, dev_boxes, dev_query_boxes, dev_iou, criterion=-1):
    """
    Args:
        N: gt数量
        K: dt数量
        dev_boxes: 一个part中的gt[x,z,dx,dz,angle](clockwise when positive) [N, 5] 
        dev_query_boxes:一个part中的dt
        dev_iou: (N,K)
        criterion:-1
    """
    threadsPerBlock = 8 * 8
    # 行列起始点
    row_start = cuda.blockIdx.x
    col_start = cuda.blockIdx.y
    # 块内线程索引
    tx = cuda.threadIdx.x
    # 确定块大小，防止最后一个块的线程超过gt数量，访问到空内存
    row_size = min(N - row_start * threadsPerBlock, threadsPerBlock)
    col_size = min(K - col_start * threadsPerBlock, threadsPerBlock)
    # 分配共享内存，一个块有64个线程，每个线程读取5个数据，组成一个box
    block_boxes = cuda.shared.array(shape=(64 * 5, ), dtype=numba.float32)
    block_qboxes = cuda.shared.array(shape=(64 * 5, ), dtype=numba.float32)
    # 计算线程的全局索引，对应全局内存
    dev_query_box_idx = threadsPerBlock * col_start + tx
    dev_box_idx = threadsPerBlock * row_start + tx
    # 将数据拷贝到共享内存中
    if (tx < col_size):
        block_qboxes[tx * 5 + 0] = dev_query_boxes[dev_query_box_idx * 5 + 0]
        block_qboxes[tx * 5 + 1] = dev_query_boxes[dev_query_box_idx * 5 + 1]
        block_qboxes[tx * 5 + 2] = dev_query_boxes[dev_query_box_idx * 5 + 2]
        block_qboxes[tx * 5 + 3] = dev_query_boxes[dev_query_box_idx * 5 + 3]
        block_qboxes[tx * 5 + 4] = dev_query_boxes[dev_query_box_idx * 5 + 4]
    if (tx < row_size):
        block_boxes[tx * 5 + 0] = dev_boxes[dev_box_idx * 5 + 0]
        block_boxes[tx * 5 + 1] = dev_boxes[dev_box_idx * 5 + 1]
        block_boxes[tx * 5 + 2] = dev_boxes[dev_box_idx * 5 + 2]
        block_boxes[tx * 5 + 3] = dev_boxes[dev_box_idx * 5 + 3]
        block_boxes[tx * 5 + 4] = dev_boxes[dev_box_idx * 5 + 4]
    # 线程同步
    cuda.syncthreads()
    # 块内所以线程
    if tx < row_size:
        # 一个线程处理一个gt和col_size个dt的iou
        for i in range(col_size):
            # 这里的偏移是将iou的二维矩阵拉成一维（N*K）--> (row_start * threadsPerBlock + tx) * K  +  col_start * threadsPerBlock + i
            offset = row_start * threadsPerBlock * K + col_start * threadsPerBlock + tx * K + i
            dev_iou[offset] = devRotateIoUEval(block_qboxes[i * 5:i * 5 + 5],
                                           block_boxes[tx * 5:tx * 5 + 5], criterion)


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    """rotated box iou running in gpu. 500x faster than cpu version
    (take 5ms in one example with numba.cuda code).
    convert from [this project](
        https://github.com/hongzhenwang/RRPN-revise/tree/master/pcdet/rotation).
    
    Args:
        boxes: 一个part中gt[x,z,dx,dz,angle](clockwise when positive) [N, 5] 
        query_boxes: 一个part中dt
        device_id (int, optional): Defaults to 0. 
    
    Returns:
        iou
    """
    # 确定数据类型并转换数据类型
    box_dtype = boxes.dtype
    boxes = boxes.astype(np.float32)
    query_boxes = query_boxes.astype(np.float32)
    # 确定gt和dt的数量
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    # 初始化iou
    iou = np.zeros((N, K), dtype=np.float32)
    if N == 0 or K == 0:
        return iou
    # 确定每个block的线程数64
    threadsPerBlock = 8 * 8
    # 选择device_id=0
    cuda.select_device(device_id)
    # 计算grid中的块大小，二维，向上取整
    blockspergrid = (div_up(N, threadsPerBlock), div_up(K, threadsPerBlock))
    # 创建cuda流
    stream = cuda.stream()
    # 流同步
    with stream.auto_synchronize():
        # 将数据放到cuda上
        boxes_dev = cuda.to_device(boxes.reshape([-1]), stream)
        query_boxes_dev = cuda.to_device(query_boxes.reshape([-1]), stream)
        iou_dev = cuda.to_device(iou.reshape([-1]), stream)
        # 调用核函数
        rotate_iou_kernel_eval[blockspergrid, threadsPerBlock, stream](
            N, K, boxes_dev, query_boxes_dev, iou_dev, criterion)
        # 将结果拷贝会host
        iou_dev.copy_to_host(iou.reshape([-1]), stream=stream)
    # 返回结果
    return iou.astype(boxes.dtype)
