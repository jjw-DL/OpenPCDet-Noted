import io as sysio

import numba
import numpy as np

from .rotate_iou import rotate_iou_gpu_eval

# http://zhaoxuhui.top/blog/2019/01/17/PythonNumba.html
# https://zhuanlan.zhihu.com/p/60994299
# jit的全称是Just-in-time，在numba里面则特指Just-in-time compilation（即时编译）
# 对于Numba的@jit有两种编译模式：nopython和object模式
# nopython模式会完全编译这个被修饰的函数，函数的运行与Python解释器完全无关，不会调用Python的C语言API@njit这个装饰器与@jit(nopython=True)等价
# object模式中编译器会自动识别函数中循环语句等可以编译加速的代码部分，并编译成机器码，对于剩下不能识别的部分交给Python解释器运行
# 如果没设置参数nopython=True，Numba首先会尝试使用nopython模式，如果因为某些原因无法使用，则会使用object模式
# 加了nopython后则会强制编译器使用nopython模式，但如果代码出现了不能自动推导的类型，有报错的风险
# Numba likes loops
# Numba likes NumPy functions
# Numba likes NumPy broadcasting
@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    """
    获取对于recall离散化必须要评估的scores
    当不再满足r_recall-current_recall < current_recall-l_recall时就退出
    在退出前，每有一个满足的current_recall就加1/40 --> 数值人为规定
    返回的是排名前40的满足条件的scores
    Args:
        scores: 所有有效gt匹配dt的分数，tp的数量
        num_gt: 全部有效box的数量
    Returns：
        thresholds: 41个分界点
    """
    scores.sort() # 分数排序，从小到大
    scores = scores[::-1] # 从大到小
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt # 计算左recall
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt # 计算右recall
        else:
            r_recall = l_recall
        # 找到40个recall的分界点，current_recall在l_recall和r_recall之间
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0) # 每次增加0.25
    return thresholds


def clean_data(gt_anno, dt_anno, current_class, difficulty):
    """
    根据current_class和difficulty对gt和dt的box进行分类标注并计算有效box数量和dc_bboxes
    Args:
        gt_anno:单帧点云标注dict
        dt_anno:单帧点云预测dict
        current_class: 标量 0
        difficulty: 标量 0
    Returns:
        num_valid_gt:有效gt的数量
        ignored_gt: gt标志列表 0有效，1忽略，-1其他
        ignored_dt：dt标志列表
        dc_bboxes: don't care的box
    """
    CLASS_NAMES = ['car', 'pedestrian', 'cyclist', 'van', 'person_sitting', 'truck'] # 类别
    MIN_HEIGHT = [40, 25, 25] # 最小高度阈值
    MAX_OCCLUSION = [0, 1, 2] # 最大遮挡阈值
    MAX_TRUNCATION = [0.15, 0.3, 0.5] # 最大截断阈值
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno["name"]) # gt数量 eg:7
    num_dt = len(dt_anno["name"]) # dt数量 eg:13
    num_valid_gt = 0
    # 2.遍历所有gt框
    for i in range(num_gt):
        # 获取第i个bbox，name和height等信息
        bbox = gt_anno["bbox"][i]
        gt_name = gt_anno["name"][i].lower()
        height = bbox[3] - bbox[1]
        valid_class = -1
        # 2.1首先，根据类别进行三个判断给类别分类: 1有效，0忽略，-1无效
        if (gt_name == current_cls_name):
            valid_class = 1
        elif (current_cls_name == "Pedestrian".lower()
              and "Person_sitting".lower() == gt_name):
            valid_class = 0
        elif (current_cls_name == "Car".lower() and "Van".lower() == gt_name):
            valid_class = 0
        else:
            valid_class = -1
        ignore = False
        # 2.2然后，根据occluded，truncated和height判断该box是否要忽略
        if ((gt_anno["occluded"][i] > MAX_OCCLUSION[difficulty]) # 遮挡严重
                or (gt_anno["truncated"][i] > MAX_TRUNCATION[difficulty]) # 截断严重
                or (height <= MIN_HEIGHT[difficulty])): # 高度太小
            # if gt_anno["difficulty"][i] > difficulty or gt_anno["difficulty"][i] == -1:
            ignore = True
        # 2.3最后，根据之前得到的valid_class和ignore状态量进行分类
        if valid_class == 1 and not ignore:
            ignored_gt.append(0) # 0表示有效
            num_valid_gt += 1 # 有效框数量+1
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1) # 1表示忽略
        else:
            ignored_gt.append(-1) # -1表示无效
    # for i in range(num_gt):
        # 2.4如果类别是DontCare，则在dc_bboxes中加入该box
        if gt_anno["name"][i] == "DontCare":
            dc_bboxes.append(gt_anno["bbox"][i])
    
    # 3.遍历dt box 
    for i in range(num_dt):
        # 3.1 如果检测到的box的类比是当前的类别，则有效类别标志置1
        if (dt_anno["name"][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        # 3.2 计算高度
        height = abs(dt_anno["bbox"][i, 3] - dt_anno["bbox"][i, 1])
        # 如果检测出的高度小于最小高度，则忽略该box
        if height < MIN_HEIGHT[difficulty]:
            ignored_dt.append(1)
        # 3.3 如果有效，则加入0
        elif valid_class == 1:
            ignored_dt.append(0)
        # 否则，加入-1
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    """
    计算图像box的iou
    Args:
        boxes:一个part中的全部gt，以第一个part为例(642,4)
        query_boxes：一个part中的全部dt，以第一个part为例(233,4)
    """
    N = boxes.shape[0] # gt_box的总数
    K = query_boxes.shape[0] # det_box的总数
    # 初始化overlap矩阵
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    # 两层for循环逐个box计算iou，因为有jit加速，所以成for循环的形式
    for k in range(K):
        # 计算第k个dt box的面积（box是左上角和右下角的形式[x1,y1,x2,y2]）
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N): # 遍历gt boxes
            # 重叠部分的宽度 = 两个图像右边缘的较小值 - 两个图像左边缘的较大值
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0: # 如果宽度方向有重叠，再计算高度
                # 重叠部分的高度 = 两个图像上边缘的较小值 - 两个图像下边缘的较大值
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1: # 默认执行criterion = -1
                        # 求两个box的并
                        ua = (
                            (boxes[n, 2] - boxes[n, 0]) *
                            (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def bev_box_overlap(boxes, qboxes, criterion=-1):
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou

# 以非python方式编译，编译原生多线程
# 编译器将编译一个版本，并行运行多个原生的线程（没有GIL）
@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    """
    计算box的3D iou
    Args:
        boxes:一个part中的全部gt，以第一个part为例(642,7) [x,y,z,dx,dy,dz,alpha]
        query_boxes：一个part中的全部dt，以第一个part为例(233,7)
        rinc:在鸟瞰图视角下的iou（642，233）
    Returns:
        返回3D iou-->rinc
    """
    # ONLY support overlap in CAMERA, not lidar.
    # 在相机坐标系下进行计算，z轴朝前，y轴朝下，x轴朝右，因此，俯视图取x和z
    N, K = boxes.shape[0], qboxes.shape[0]
    # 遍历gt
    for i in range(N):
        # 遍历dt
        for j in range(K):
            # 如果鸟瞰视角存在重叠
            if rinc[i, j] > 0:
                # iw = (min(boxes[i, 1] + boxes[i, 4], qboxes[j, 1] +
                #         qboxes[j, 4]) - max(boxes[i, 1], qboxes[j, 1]))
                # 这里的1是y轴，在相机坐标系下就是高度方向，重叠高度=上边缘的最小值-下边缘的最大值
                iw = (min(boxes[i, 1], qboxes[j, 1]) - max(
                    boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4]))
                # 如果重叠高度 > 0
                if iw > 0:
                    # 1.求两个box的体积
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    # 2.求交集体积
                    inc = iw * rinc[i, j]
                    # 3.根据criterion，计算并集体积
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    # 4.计算交并比
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1):
    """
    计算鸟瞰图视角下box的iou（带旋转角）
    Args:
        boxes:一个part中的全部gt，以第一个part为例(642,7) [x,y,z,dx,dy,dz,alpha]
        query_boxes：一个part中的全部dt，以第一个part为例(233,7)
        rinc:在鸟瞰图视角下的iou（642，233）
    Returns:
        返回3D iou-->rinc
    """
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]],
                               qboxes[:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


@numba.jit(nopython=True)
def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           ignored_gt,
                           ignored_det,
                           dc_bboxes,
                           metric,
                           min_overlap,
                           thresh=0,
                           compute_fp=False,
                           compute_aos=False):
    """
    逐帧计算tp, fp, fn, similarity, thresholds[:thresh_idx]等统计指标
    根据compute_fp的状态不同，存在两种模式
    Args:
        overlaps:单帧点云的iou（N,M)\n
        gt_datas:(N,5)--> (x1, y1, x2, y2, alpha)\n
        dt_datas:(M,6)--> (x1, y1, x2, y2, alpha, score)\n
        ignored_gt:（N,）为box的状态 0，1，-1\n
        ignored_det:（M,）为box的状态 0，1，-1\n
        dc_bboxes:(k,4)\n
        metric:0: bbox, 1: bev, 2: 3d\n
        min_overlap:最小iou阈值\n
        thresh=0:忽略score低于此值的dt，根据recall点会传入41个阈值\n
        compute_fp=False\n
        compute_aos=False\n
    Returns:
        tp: 真正例 预测为真，实际为真\n
        fp：假正例 预测为真，实际为假\n
        fn：假负例 预测为假，实际为真\n
        similarity:余弦相似度\n
        thresholds[:thresh_idx]:与有效gt匹配dt分数\n
    precision = TP / TP + FP 所有预测为真中，TP的比重\n
    recall = TP / TP + FN 所有真实为真中，TP的比重\n
    """
    # ============================ 1 初始化============================
    det_size = dt_datas.shape[0] # det box的数量 M
    gt_size = gt_datas.shape[0] # gt box的数量 N 
    dt_scores = dt_datas[:, -1] # dt box的得分 (M,)
    dt_alphas = dt_datas[:, 4] # dt alpha的得分 (M,)
    gt_alphas = gt_datas[:, 4] # gt alpha的得分 (N,)
    dt_bboxes = dt_datas[:, :4] # (M,4)
    gt_bboxes = gt_datas[:, :4] # (N,4)

    # 该处的初始化针对dt
    assigned_detection = [False] * det_size # 存储dt是否匹配了gt
    ignored_threshold = [False] * det_size # 如果dt分数低于阈值，则标记为True

    # 如果计算fp: 预测为真，实际为假
    if compute_fp:
        # 遍历dt 
        for i in range(det_size):
            # 如果分数低于阈值
            if (dt_scores[i] < thresh):
                # 忽略该box
                ignored_threshold[i] = True

    # 初始化
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    # thresholds = [0.0]
    # delta = [0.0]
    thresholds = np.zeros((gt_size, )) #（N,）
    thresh_idx = 0 # thresholds的index，后面更新
    delta = np.zeros((gt_size, )) #（N,）
    delta_idx = 0 # delta的index，后面更新

    # ============================ 2 针对gt找匹配的dt，同时计算tp和fn，因为这是针对gt的============================
    # 遍历gt，这里还是以gt为主，针对gt找匹配的dt，跳过不合格的gt的dt
    for i in range(gt_size):
        if ignored_gt[i] == -1: # 跳过无效gt
            continue
        det_idx = -1 # 储存目前为止匹配dt的最佳idx
        valid_detection = NO_DETECTION # 标记是否为有效dt
        max_overlap = 0 # 存储到目前为止匹配dt的最佳overlap
        assigned_ignored_det = False # 标记是否匹配上dt

        # 遍历dt
        for j in range(det_size):
            if (ignored_det[j] == -1): # 跳过无效dt
                continue
            if (assigned_detection[j]): # 如果已经匹配了gt，跳过
                continue 
            if (ignored_threshold[j]): # 如果dt分数低于阈值，则跳过
                continue

            overlap = overlaps[j, i] # 获取当前dt和此gt之间的overlap 
            dt_score = dt_scores[j] # 获取当前dt的分数

            if (not compute_fp # compute_fp为false，不需要计算FP
                    and (overlap > min_overlap) # overlap大于最小阈值比如0.7
                    and dt_score > valid_detection): # 找最高分的检测
                det_idx = j
                valid_detection = dt_score # 更新到目前为止检测到的最大分数
            elif (compute_fp # 当compute_fp为true时，基于overlap进行选择
                  and (overlap > min_overlap) # overlap要大于最小值
                  and (overlap > max_overlap or assigned_ignored_det) # 如果当前overlap比之前的最大overlap还大或者gt以及匹配dt
                  and ignored_det[j] == 0): # dt有效
                max_overlap = overlap # 更新最佳的overlap
                det_idx = j # 更新最佳匹配dt的id
                valid_detection = 1 # 标记有效dt
                assigned_ignored_det = False # 用留一法来表明已经分配了关心的单位
            elif (compute_fp # compute_fp为true
                  and (overlap > min_overlap) # 如果重叠足够
                  and (valid_detection == NO_DETECTION) # 尚未分配任何东西
                  and ignored_det[j] == 1): # dt被忽略
                det_idx = j  # 更新最佳匹配dt的id
                valid_detection = 1 # 标记有效dt
                assigned_ignored_det = True # 标志gt已经匹配上dt

        # 如果有效gt没有找到匹配，则fn加一，因为真实标签没有找到匹配
        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        # 如果gt找到了匹配，并且gt标志为忽略或者dt标志为忽略，则assigned_detection标记为True
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        # 否则有效gt找到了匹配
        elif valid_detection != NO_DETECTION:
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx] # 将该dt的分数赋予thresholds
            thresh_idx += 1
            if compute_aos:
                # delta.append(gt_alphas[i] - dt_alphas[det_idx])
                delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx] # 两个alpha相减，为delta
                delta_idx += 1

            assigned_detection[det_idx] = True
    
    # ============================ 3 计算fp，这是针对dt的 ============================
    # 在遍历完全部的gt和dt后，如果compute_fp为真，则计算fp
    if compute_fp:
        # 遍历dt
        for i in range(det_size):
            # 如果以下四个条件全部为false，则fp加1
            # assigned_detection[i] == 0 --> dt没有分配gt
            # ignored_det[i] != -1 and ignored_det[i] ！= 1 --> gnored_det[i] == 0 有效dt
            # ignored_threshold[i] == false, 无法忽略该dt box
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1 # 预测为真，实际为假
        
        nstuff = 0 # don't care的数量
        if metric == 0: # 如果计算的是bbox
            # 计算dt和dc的iou，并且criterion = 0，并集就是dt的面积
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
            for i in range(dc_bboxes.shape[0]):
                for j in range(det_size):
                    # 跳过上面没有添加到fp的内容
                    if (assigned_detection[j]):
                        continue
                    if (ignored_det[j] == -1 or ignored_det[j] == 1):
                        continue
                    if (ignored_threshold[j]):
                        continue
                    # 如果两者之间的重叠大于min_overlap
                    if overlaps_dt_dc[j, i] > min_overlap:
                        assigned_detection[j] = True # 将检测结果分配给dt
                        nstuff += 1 
        fp -= nstuff # 从之前计算的fp中去除don't care，don't care不参与计算

        if compute_aos:
            # fp+tp(有效gt找到了匹配）= 在该recall阈值下的所有预测为真的box数量
            tmp = np.zeros((fp + delta_idx, ))
            # tmp = [0] * fp
            # 相似度计算公式
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
                # tmp.append((1.0 + np.cos(delta[i])) / 2.0)
            # assert len(tmp) == fp + tp
            # assert len(delta) == tp
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp) # 这里直接累加个数，在eval_class中在进行除法
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]


def get_split_parts(num, num_part): # num:3769 num_part:100
    """
    返回数字列表，将num分为num_part个，其余部分在末尾
    """
    same_part = num // num_part # 37
    remain_num = num % num_part # 69
    if same_part == 0:
        return [num]

    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num] # [37] * 100 + [69] --> [37 37 37 .... 69]


@numba.jit(nopython=True)
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             dc_nums,
                             gt_datas,
                             dt_datas,
                             dontcares,
                             ignored_gts,
                             ignored_dets,
                             metric,
                             min_overlap,
                             thresholds,
                             compute_aos=False):
    """
    计算part的pr
    Args:
        overlaps: 一个part的iou (M,N)-->(642,233)
        pr: （41，4）--> tp, fp, fn, similarity
        gt_nums: 一个part的gt的数量 （37，）
        dt_nums: 一个part的dt的数量 （37，）
        dc_nums: 一个part的dc的数量 （37，）
        gt_datas: 一个part的gt的数据 (233,5)
        dt_datas: 一个part的gt的数据 (642,5)
        dontcares: 一个part的gt的数据 (79,4)
        ignored_gts: (233,)
        ignored_dets: (642,)
        metric: 0
        min_overlap: 0.7
        thresholds: (41,)
        compute_aos=False: True
    Return：
        传入的参数有pr，因此没有该函数没有返回值，返回值在参数中
    """
    gt_num = 0
    dt_num = 0
    dc_num = 0
    # 遍历part点云，逐帧计算累加pr和box数量
    for i in range(gt_nums.shape[0]):
        # 遍历阈值
        for t, thresh in enumerate(thresholds):
            # 提取该帧点云的iou矩阵
            overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:
                               gt_num + gt_nums[i]] # （13，7）
            # 取出该帧的数据
            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]] #（7，5）
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]] #（13，6）
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]] # （7，）
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]] # （13，）
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]] # (4, 4)
            # 真正计算指标
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                dontcare,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos)
            # 累加计算指标
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        # 累加box数量
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50):
    """fast iou algorithm. this function can be used independently to
    do result analysis. Must be used in CAMERA coordinate system.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        num_parts: int. a parameter for fast calculate algorithm
    """
    assert len(gt_annos) == len(dt_annos)
    # 1.计算每一帧点云的box数量
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0) # （3769，）[7, 2, 7...]
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0) # （3769，）[13,13,11 ...]
    num_examples = len(gt_annos) # 3769
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0
    # 2.按照part计算指标
    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part] # 取出37帧点云标注信息
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        # 根据metric的index分别计算不同指标
        if metric == 0:
            gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0) # 将box信息进行拼接，以第一个part为例(642,4)
            dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0) # (233, 4)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        # 在相机坐标系下进行计算，z轴朝前，y轴朝下，x轴朝右，因此，俯视图取x和z
        elif metric == 1:
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in gt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1) # （N, 5）
            loc = np.concatenate(
                [a["location"][:, [0, 2]] for a in dt_annos_part], 0)
            dims = np.concatenate(
                [a["dimensions"][:, [0, 2]] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1) # （K, 5）
            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(
                np.float64)
        elif metric == 2:
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(
                np.float64)
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part) # 记录part的overlaps，共101个元素
        example_idx += num_part # 更新example_idx，作为新的part的起点
    # 前面进行了多余的计算（一个part内的不同点云之间也计算了overlap），这里进行截取    
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part] # 37
        dt_annos_part = dt_annos[example_idx:example_idx + num_part] # 37
        gt_num_idx, dt_num_idx = 0, 0
        # 逐帧计算overlaps
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            # 在overlaps的第j个part中根据gt_box_num和dt_box_num截取对应的iou
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part
    # overlaps共3769个元素，每个元素代表了一帧点云的iou
    # parted_overlaps共101个元素，表示每个part的iou
    # total_gt_num共3769个元素，表示每帧点云的gt_box的数量
    # total_dt_num共3769个元素，表示每帧点云的det_box的数量
    return overlaps, parted_overlaps, total_gt_num, total_dt_num


def _prepare_data(gt_annos, dt_annos, current_class, difficulty):
    """
    进行数据分类和整合
    Args：
        gt_annos:点云标注dict 3796
        dt_annos:点云预测dict 3796
        current_class: 标量 0
        difficulty: 标量 0
    Returns:
        gt_datas_list: 3769个元素，每个元素为(N,5)--> (x1, y1, x2, y2, alpha) 注意N不相等
        dt_datas_list: 3769个元素，每个元素为(M,6)--> (x1, y1, x2, y2, alpha, score) 注意M不相等
        ignored_gts: 3769个元素,每个元素为（N,）为每个box的状态 0，1，-1
        ignored_dets: 3769个元素,每个元素为（M,）为每个box的状态 0，1，-1
        dontcares: 3769个元素，每个元素为(k,4) 注意K不相等
        total_dc_num: 3769个元素，表示每帧点云dc box的数量
        total_num_valid_gt:全部有效box的数量: 2906
    """
    # 数据初始化
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0

    # 遍历测试集点云
    for i in range(len(gt_annos)):
        # 清洗数据
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty)
        # num_valid_gt: 有效gt的数量
        # ignored_gt: gt标志列表 0有效，1忽略，-1其他
        # ignored_dt：dt标志列表
        # dc_bboxes: Don't Care的box
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets

        # 将ignored_gt状态列表加入ignored_gts列表
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))

        # 处理don't care的box
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0]) #全部don't care box的数量
        dontcares.append(dc_bboxes) # 将don't care的box加入列表
        total_num_valid_gt += num_valid_gt # 计算总的有效gt的数量
        # 将检测结果从新组合拼接
        gt_datas = np.concatenate(
            [gt_annos[i]["bbox"], gt_annos[i]["alpha"][..., np.newaxis]], 1) # (N, 5) --> (x1, y1, x2, y2, alpha)
        dt_datas = np.concatenate([
            dt_annos[i]["bbox"], dt_annos[i]["alpha"][..., np.newaxis],
            dt_annos[i]["score"][..., np.newaxis]
        ], 1) # （M, 6）--> (x1, y1, x2, y2, alpha, score)
        # 将结果加入list中
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0) # (3769,) 全部don't care box的数量列表
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
            total_dc_num, total_num_valid_gt)


def eval_class(gt_annos,
               dt_annos,
               current_classes,
               difficultys,
               metric,
               min_overlaps,
               compute_aos=False,
               num_parts=100):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        current_classes: list of int, 0: car, 1: pedestrian, 2: cyclist
        difficultys: list of int. eval difficulty, 0: easy, 1: normal, 2: hard
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        min_overlaps: float, min overlap. format: [num_overlap, metric, class]. （2，3，3）
        num_parts: int. a parameter for fast calculate algorithm

    Returns:
        dict of recall, precision and aos
    """
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos) # 3769
    # 1、split_parts共101个元素，100个37和一个69--> [37,37......69]
    split_parts = get_split_parts(num_examples, num_parts)
    # 2、计算iou
    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    # overlaps共3769个元素，每个元素代表了一帧点云的iou
    # parted_overlaps共101个元素，表示每个part的iou
    # total_gt_num共3769个元素，表示每帧点云的gt_box的数量
    # total_dt_num共3769个元素，表示每帧点云的det_box的数量
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets

    # 3、计算初始化需要的数组维度
    N_SAMPLE_PTS = 41
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    num_difficulty = len(difficultys)
    # 初始化precision,recall和aos
    precision = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])

    # 4、逐类别遍历 0: car, 1: pedestrian, 2: cyclist
    for m, current_class in enumerate(current_classes):
        # 逐难度遍历 0: easy, 1: normal, 2: hard
        for l, difficulty in enumerate(difficultys):
            # 4.1 数据准备
            rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty)
            # gt_datas_list: 3769个元素，每个元素为(N,5)--> (x1, y1, x2, y2, alpha) 注意N不相等
            # dt_datas_list: 3769个元素，每个元素为(M,6)--> (x1, y1, x2, y2, alpha, score) 注意M不相等
            # ignored_gts: 3769个元素,每个元素为（N,）为每个box的状态 0，1，-1
            # ignored_dets: 3769个元素,每个元素为（M,）为每个box的状态 0，1，-1
            # dontcares: 3769个元素，每个元素为(k,4) 注意K不相等
            # total_dc_num: 3769个元素，表示每帧点云dc box的数量
            # total_num_valid_gt:全部有效box的数量: 2906
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
             dontcares, total_dc_num, total_num_valid_gt) = rets
            # 4.2 逐min_overlap遍历
            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                thresholdss = []
                # 4.2.1 计算所有有效gt匹配上的dt的全部分数，共2838个box score和41个recall阈值点
                for i in range(len(gt_annos)):
                    # 这里调用compute_statistics_jit没有真正计算指标，而是获取thresholdss，然后计算41个recall阈值点
                    rets = compute_statistics_jit(
                        overlaps[i], # 单帧点云的iou（N,M）
                        gt_datas_list[i], # (N,5)--> (x1, y1, x2, y2, alpha)
                        dt_datas_list[i], # (M,6)--> (x1, y1, x2, y2, alpha, score)
                        ignored_gts[i], # （N,）为box的状态 0，1，-1
                        ignored_dets[i], # （M,）为box的状态 0，1，-1
                        dontcares[i], # (k,4)
                        metric, # 0: bbox, 1: bev, 2: 3d
                        min_overlap=min_overlap, # 最小iou阈值
                        thresh=0.0, # 忽略score低于此值的dt
                        compute_fp=False)
                    tp, fp, fn, similarity, thresholds = rets
                    thresholdss += thresholds.tolist() # list的加法和extend类似(2838,)
                thresholdss = np.array(thresholdss)
                thresholds = get_thresholds(thresholdss, total_num_valid_gt) # 获取41个recall阈值点
                thresholds = np.array(thresholds)
                pr = np.zeros([len(thresholds), 4]) # (41, 4)
                idx = 0

                # 4.2.2 遍历101个part，在part内逐帧逐recall_threshold计算tp, fp, fn, similarity，累计pr
                for j, num_part in enumerate(split_parts):
                    gt_datas_part = np.concatenate(
                        gt_datas_list[idx:idx + num_part], 0) # (N,5)
                    dt_datas_part = np.concatenate(
                        dt_datas_list[idx:idx + num_part], 0) # (M,6)
                    dc_datas_part = np.concatenate(
                        dontcares[idx:idx + num_part], 0) # (K, 4)
                    ignored_dets_part = np.concatenate(
                        ignored_dets[idx:idx + num_part], 0) # (M,)
                    ignored_gts_part = np.concatenate(
                        ignored_gts[idx:idx + num_part], 0) # (N,)
                    # 真正计算指标，融合统计结果
                    fused_compute_statistics(
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx:idx + num_part],
                        total_dt_num[idx:idx + num_part],
                        total_dc_num[idx:idx + num_part],
                        gt_datas_part,
                        dt_datas_part,
                        dc_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        metric,
                        min_overlap=min_overlap,
                        thresholds=thresholds,
                        compute_aos=compute_aos)
                    idx += num_part
                
                # 4.2.3 根据不同类别，难度和最小iou阈值以及recall阈值，计算指标
                for i in range(len(thresholds)):
                    # m:类比，l:难度，k:min_overlap, i:threshold
                    # pr:（41，4）--> tp, fp, fn, similarity
                    recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2]) # recall = tp / (tp + fn) 真实值 （3，3，2，41）
                    precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1]) # precision = tp / (tp + fp) 预测值 （3，3，2，41）
                    if compute_aos:
                        aos[m, l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1]) # aos = similarity / (tp + fp) 
                # 4.2.4 因为pr曲线是外弧形，按照threshold取该节点后面的最大值，相当于按照节点截取矩形
                for i in range(len(thresholds)):
                    precision[m, l, k, i] = np.max(
                        precision[m, l, k, i:], axis=-1)
                    recall[m, l, k, i] = np.max(recall[m, l, k, i:], axis=-1)
                    if compute_aos:
                        aos[m, l, k, i] = np.max(aos[m, l, k, i:], axis=-1)
    ret_dict = {
        "recall": recall, # （3，3，2，41）
        "precision": precision, # （3，3，2，41）
        "orientation": aos, # （3，3，2，41）
    }
    return ret_dict


def get_mAP(prec):
    """
    map:pr曲线的面积，按照分的份数区分mAP_R11和mAP_R40
    """
    sums = 0
    # 每隔四个取一个点，假设宽度为1，最后归一化
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100 


def get_mAP_R40(prec):
    sums = 0
    for i in range(1, prec.shape[-1]):
        sums = sums + prec[..., i]
    return sums / 40 * 100


def print_str(value, *arg, sstream=None):
    "结果打印函数"
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()


def do_eval(gt_annos,
            dt_annos,
            current_classes,
            min_overlaps,
            compute_aos=False,
            PR_detail_dict=None):
    """
    计算评价指标
    Args:
        gt_annos:3769帧的测试点云标注字典，包括{name,alpah,rotation_y,gt_boxes_lidar}等信息
        dt_annos:3769帧的测试点云预测字典，包括{name,alpah,rotation_y,boxes_lidar}等信息
        current_classes:[0,1,2]
        min_overlaps：(2,3,3)
        compute_aos:True
        PR_detail_dict:None
    """
    # min_overlaps: [num_minoverlap, metric, num_class]
    difficultys = [0, 1, 2] # difficultys: list of int. eval difficulty, 0: easy, 1: normal, 2: hard
    # bbox和bev，3d以及aos的计算相似，都是调用eval_class，只是第五个参数分别是0，1，2
    # ret_dict = {
    #     "recall": （3，3，2，41）
    #     "precision":（3，3，2，41）
    #     "orientation":（3，3，2，41）
    # } m:类比，l:难度，k:min_overlap, i:threshold
    # ret: [num_class, num_diff, num_minoverlap, num_sample_points]
    # eval_class会根据不同的metric调用calculate_iou_partly计算不同的iou
    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 0,
                     min_overlaps, compute_aos)
    
    mAP_bbox = get_mAP(ret["precision"])
    mAP_bbox_R40 = get_mAP_R40(ret["precision"])

    if PR_detail_dict is not None:
        PR_detail_dict['bbox'] = ret['precision']

    mAP_aos = mAP_aos_R40 = None
    if compute_aos:
        mAP_aos = get_mAP(ret["orientation"])
        mAP_aos_R40 = get_mAP_R40(ret["orientation"])

        if PR_detail_dict is not None:
            PR_detail_dict['aos'] = ret['orientation']

    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 1,
                     min_overlaps)
    mAP_bev = get_mAP(ret["precision"])
    mAP_bev_R40 = get_mAP_R40(ret["precision"])

    if PR_detail_dict is not None:
        PR_detail_dict['bev'] = ret['precision']

    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 2,
                     min_overlaps)
    mAP_3d = get_mAP(ret["precision"])
    mAP_3d_R40 = get_mAP_R40(ret["precision"])
    if PR_detail_dict is not None:
        PR_detail_dict['3d'] = ret['precision']
    # 返回8个值，分别是mAP和mAP_R40的bbox，bev，3d和aos等信息
    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos, mAP_bbox_R40, mAP_bev_R40, mAP_3d_R40, mAP_aos_R40


def do_coco_style_eval(gt_annos, dt_annos, current_classes, overlap_ranges,
                       compute_aos):
    # overlap_ranges: [range, metric, num_class]
    min_overlaps = np.zeros([10, *overlap_ranges.shape[1:]])
    for i in range(overlap_ranges.shape[1]):
        for j in range(overlap_ranges.shape[2]):
            min_overlaps[:, i, j] = np.linspace(*overlap_ranges[:, i, j])
    mAP_bbox, mAP_bev, mAP_3d, mAP_aos = do_eval(
        gt_annos, dt_annos, current_classes, min_overlaps, compute_aos)
    # ret: [num_class, num_diff, num_minoverlap]
    mAP_bbox = mAP_bbox.mean(-1)
    mAP_bev = mAP_bev.mean(-1)
    mAP_3d = mAP_3d.mean(-1)
    if mAP_aos is not None:
        mAP_aos = mAP_aos.mean(-1)
    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos


def get_official_eval_result(gt_annos, dt_annos, current_classes, PR_detail_dict=None):
    """
    计算指标最外层函数（类似main函数），同时负责打印和输出
    Args:
        gt_annos:3769帧的测试点云标注字典，包括{name,alpah,rotation_y,gt_boxes_lidar}等信息
        dt_annos:3769帧的测试点云预测字典，包括{name,alpah,rotation_y,boxes_lidar}等信息
        current_classes:[0,1,2]
        compute_aos:True
        PR_detail_dict:None
    """
    # 给不同的类别设置不同的阈值
    # 6类在easy、moderate和hard的iou阈值
    overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7], 
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7]])
    overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5], 
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])
    # 每个类别输出4个结果
    # 分别采用AP在overlap=0.7或0.5的结果
    # 和采用AP_R40在overlap=0.7或0.5的结果
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)  
    class_to_name = {
        0: 'Car', 			#[[0.7,0.7,0.7],[0.7, 0.5, 0.5]]
        1: 'Pedestrian',	#[[0.5,0.5,0.5],[0.5,0.25,0.25]]
        2: 'Cyclist',		#[[0.5,0.5,0.5],[0.5,0.25,0.25]]
        3: 'Van',			#[[0.7,0.7,0.7],[0.7, 0.5, 0.5]]
        4: 'Person_sitting',#[[0.5,0.5,0.5],[0.5,0.25,0.25]]
        5: 'Truck'			#[[0.7,0.7,0.7],[0.5, 0.5, 0.5]]
    }
    name_to_class = {v: n for n, v in class_to_name.items()} # 将类别名和数字对应
    # 如果传入的类比不是列表则转换为list
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    # 将需要评估的类比转换为数字
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    # 根据类别在min_overlaps选择iou阈值
    min_overlaps = min_overlaps[:, :, current_classes] # 取前3列
    result = ''
    # check whether alpha is valid，决定是否计算AOS
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break
    
    # 计算结果的核心函数：传入的gt和det annos以及检测类别，iou阈值和是否计算AOS标志，PR_detail_dict=None
    mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = do_eval(
        gt_annos, dt_annos, current_classes, min_overlaps, compute_aos, PR_detail_dict=PR_detail_dict)

    # 打印结果
    ret_dict = {}
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        for i in range(min_overlaps.shape[0]):
            result += print_str(
                (f"{class_to_name[curcls]} "
                 "AP@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j]))) # i代表选择iou 0.7还是0.5，j选择类别
            
            result += print_str((f"bbox AP:{mAPbbox[j, 0, i]:.4f}, "
                                 f"{mAPbbox[j, 1, i]:.4f}, "
                                 f"{mAPbbox[j, 2, i]:.4f}"))
            result += print_str((f"bev  AP:{mAPbev[j, 0, i]:.4f}, "
                                 f"{mAPbev[j, 1, i]:.4f}, "
                                 f"{mAPbev[j, 2, i]:.4f}"))
            result += print_str((f"3d   AP:{mAP3d[j, 0, i]:.4f}, "
                                 f"{mAP3d[j, 1, i]:.4f}, "
                                 f"{mAP3d[j, 2, i]:.4f}"))
            # 计算AOS
            if compute_aos:
                result += print_str((f"aos  AP:{mAPaos[j, 0, i]:.2f}, "
                                     f"{mAPaos[j, 1, i]:.2f}, "
                                     f"{mAPaos[j, 2, i]:.2f}"))
                # if i == 0:
                   # ret_dict['%s_aos/easy' % class_to_name[curcls]] = mAPaos[j, 0, 0]
                   # ret_dict['%s_aos/moderate' % class_to_name[curcls]] = mAPaos[j, 1, 0]
                   # ret_dict['%s_aos/hard' % class_to_name[curcls]] = mAPaos[j, 2, 0]
            
            # AP_R40 输出到控制台，并记录到ret_dict中，写入文件
            result += print_str(
                (f"{class_to_name[curcls]} "
                 "AP_R40@{:.2f}, {:.2f}, {:.2f}:".format(*min_overlaps[i, :, j])))
            result += print_str((f"bbox AP:{mAPbbox_R40[j, 0, i]:.4f}, "
                                 f"{mAPbbox_R40[j, 1, i]:.4f}, "
                                 f"{mAPbbox_R40[j, 2, i]:.4f}"))
            result += print_str((f"bev  AP:{mAPbev_R40[j, 0, i]:.4f}, "
                                 f"{mAPbev_R40[j, 1, i]:.4f}, "
                                 f"{mAPbev_R40[j, 2, i]:.4f}"))
            result += print_str((f"3d   AP:{mAP3d_R40[j, 0, i]:.4f}, "
                                 f"{mAP3d_R40[j, 1, i]:.4f}, "
                                 f"{mAP3d_R40[j, 2, i]:.4f}"))
            if compute_aos:
                result += print_str((f"aos  AP:{mAPaos_R40[j, 0, i]:.2f}, "
                                     f"{mAPaos_R40[j, 1, i]:.2f}, "
                                     f"{mAPaos_R40[j, 2, i]:.2f}"))
                # 只记录iou=0.7的结果 i = 0
                if i == 0:
                   ret_dict['%s_aos/easy_R40' % class_to_name[curcls]] = mAPaos_R40[j, 0, 0]
                   ret_dict['%s_aos/moderate_R40' % class_to_name[curcls]] = mAPaos_R40[j, 1, 0]
                   ret_dict['%s_aos/hard_R40' % class_to_name[curcls]] = mAPaos_R40[j, 2, 0]

            if i == 0:
                ret_dict['%s_3d/easy_R40' % class_to_name[curcls]] = mAP3d_R40[j, 0, 0]
                ret_dict['%s_3d/moderate_R40' % class_to_name[curcls]] = mAP3d_R40[j, 1, 0]
                ret_dict['%s_3d/hard_R40' % class_to_name[curcls]] = mAP3d_R40[j, 2, 0]
                ret_dict['%s_bev/easy_R40' % class_to_name[curcls]] = mAPbev_R40[j, 0, 0]
                ret_dict['%s_bev/moderate_R40' % class_to_name[curcls]] = mAPbev_R40[j, 1, 0]
                ret_dict['%s_bev/hard_R40' % class_to_name[curcls]] = mAPbev_R40[j, 2, 0]
                ret_dict['%s_image/easy_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 0, 0]
                ret_dict['%s_image/moderate_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 1, 0]
                ret_dict['%s_image/hard_R40' % class_to_name[curcls]] = mAPbbox_R40[j, 2, 0]

    return result, ret_dict


def get_coco_eval_result(gt_annos, dt_annos, current_classes):
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
    }
    class_to_range = {
        0: [0.5, 0.95, 10],
        1: [0.25, 0.7, 10],
        2: [0.25, 0.7, 10],
        3: [0.5, 0.95, 10],
        4: [0.25, 0.7, 10],
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    overlap_ranges = np.zeros([3, 3, len(current_classes)])
    for i, curcls in enumerate(current_classes):
        overlap_ranges[:, :, i] = np.array(
            class_to_range[curcls])[:, np.newaxis]
    result = ''
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break
    mAPbbox, mAPbev, mAP3d, mAPaos = do_coco_style_eval(
        gt_annos, dt_annos, current_classes, overlap_ranges, compute_aos)
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        o_range = np.array(class_to_range[curcls])[[0, 2, 1]]
        o_range[1] = (o_range[2] - o_range[0]) / (o_range[1] - 1)
        result += print_str((f"{class_to_name[curcls]} "
                             "coco AP@{:.2f}:{:.2f}:{:.2f}:".format(*o_range)))
        result += print_str((f"bbox AP:{mAPbbox[j, 0]:.2f}, "
                             f"{mAPbbox[j, 1]:.2f}, "
                             f"{mAPbbox[j, 2]:.2f}"))
        result += print_str((f"bev  AP:{mAPbev[j, 0]:.2f}, "
                             f"{mAPbev[j, 1]:.2f}, "
                             f"{mAPbev[j, 2]:.2f}"))
        result += print_str((f"3d   AP:{mAP3d[j, 0]:.2f}, "
                             f"{mAP3d[j, 1]:.2f}, "
                             f"{mAP3d[j, 2]:.2f}"))
        if compute_aos:
            result += print_str((f"aos  AP:{mAPaos[j, 0]:.2f}, "
                                 f"{mAPaos[j, 1]:.2f}, "
                                 f"{mAPaos[j, 2]:.2f}"))
    return result
