import logging
import os
import pickle
import random
import shutil
import subprocess

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def check_numpy_to_torch(x):
    # 检测输入数据是否是numpy格式，如果是，则转换为torch格式
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def limit_period(val, offset=0.5, period=np.pi):
    # 首先，numpy格式数据转换为torch格式
    val, is_numpy = check_numpy_to_torch(val)
    # 将方位角限制在[-pi, pi]
    ans = val - torch.floor(val / period + offset) * period
    return ans.numpy() if is_numpy else ans


def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    # 首先利用torch.from_numpy().float将numpy转化为torch
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)
    # 构造旋转矩阵batch个
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    # 对点云坐标进行旋转
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1) # 将旋转后的点云与原始点云拼接
    return points_rot.numpy() if is_numpy else points_rot # 将点云转化为numpy格式，并返回


def mask_points_by_range(points, limit_range):
    # 根据点云的范围产生mask，过滤点云
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    return mask


def get_voxel_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:
        米制中心形式的voxle中心坐标
    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times # voxel坐标乘下采样倍数，还原到原始状态
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float() # 提取点云范围的前3维
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range # （voxel坐标+半个voxel）* voxel_size恢复米级中心坐标，再加上点云范围前3维进行平移，平移到中心形式
    return voxel_centers


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    # https://www.cnblogs.com/yyds/p/6901864.html
    # 初始化日志记录器：logger = logging.getLogger(logger_name)
    # 当直接执行一段脚本的时候，这段脚本的 __name__变量等于 '__main__'
    # 当这段脚本被导入其他程序的时候，__name__ 变量等于脚本本身的名字
    # 日志器（logger）是入口，真正干活儿的是处理器（handler），处理器（handler）还可以
    # 通过过滤器（filter）和格式器（formatter）对要输出的日志内容做过滤和格式化等处理操作
    logger = logging.getLogger(__name__) 
    logger.setLevel(log_level if rank == 0 else 'ERROR') # 设置日志级别为INFO，即只有日志级别大于等于INFO的日志才会输出
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s') # 设置日志格式
    # handler：将日志记录（log record）发送到合适的目的地（destination），比如文件，socket和控制台等。
    # 一个logger对象可以通过addHandler方法添加0到多个handler，每个handler又可以定义不同日志级别，以实现日志分级过滤显示。
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file) # 将日志消息发送到文件
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def set_random_seed(seed):
    random.seed(seed) # random库的随机种子
    np.random.seed(seed) # numpy的随机种子
    torch.manual_seed(seed) # torch的随机种子
    torch.backends.cudnn.deterministic = True # torch后端是否确定
    torch.backends.cudnn.benchmark = False # torch后端的benchmark


def get_pad_params(desired_size, cur_size):
    """
    Get padding parameters for np.pad function
    Args:
        desired_size: int, Desired padded output size
        cur_size: int, Current size. Should always be less than or equal to desired_size
    Returns:
        pad_params: tuple(int), Number of values padded to the edges (before, after)
    """
    assert desired_size >= cur_size

    # Calculate amount to pad
    diff = desired_size - cur_size
    pad_params = (0, diff)

    return pad_params


def keep_arrays_by_name(gt_names, used_classes):
    # 通过类别获取被保留的索引
    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


def init_dist_slurm(tcp_port, local_rank, backend='nccl'):
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        tcp_port:
        backend:

    Returns:

    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    rank = dist.get_rank()
    return total_gpus, rank


def init_dist_pytorch(tcp_port, local_rank, backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    num_gpus = torch.cuda.device_count() # 获取GPU数量
    torch.cuda.set_device(local_rank % num_gpus) # 设置线程
    dist.init_process_group(
        backend=backend,
        init_method='tcp://127.0.0.1:%d' % tcp_port,
        rank=local_rank,
        world_size=num_gpus
    ) # 初始化组
    rank = dist.get_rank()
    return num_gpus, rank


def get_dist_info():
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank() # 进程序号，用于进程间的通讯（rank=0 的主机为 master 节点）
        world_size = dist.get_world_size() # 获取全局进程数
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def merge_results_dist(result_part, size, tmpdir):
    rank, world_size = get_dist_info()
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    pickle.dump(result_part, open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(rank)), 'wb'))
    dist.barrier()

    if rank != 0:
        return None

    part_list = []
    for i in range(world_size):
        part_file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
        part_list.append(pickle.load(open(part_file, 'rb')))

    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results


def scatter_point_inds(indices, point_inds, shape):
    ret = -1 * torch.ones(*shape, dtype=point_inds.dtype, device=point_inds.device) # 初始化结果 (8, 21, 800, 704)
    ndim = indices.shape[-1] # 获取坐标维度 4
    flattened_indices = indices.view(-1, ndim) # 将坐标展平 (204916, 4)
    # 以下两步是经典操作
    slices = [flattened_indices[:, i] for i in range(ndim)] # 分成4个list
    ret[slices] = point_inds # 将voxle的索引写入对应位置
    return ret


def generate_voxel2pinds(sparse_tensor):
    """
    计算有效voxle在原始空间shape中的索引
    """
    device = sparse_tensor.indices.device # 获取device
    batch_size = sparse_tensor.batch_size # 获取batch_size
    spatial_shape = sparse_tensor.spatial_shape # 获取空间形状 (21, 800, 704)
    indices = sparse_tensor.indices.long() # 获取索引
    point_indices = torch.arange(indices.shape[0], device=device, dtype=torch.int32) # 生成索引 (204916,)
    output_shape = [batch_size] + list(spatial_shape) # 计算输出形状 (8, 21, 800, 704)
    v2pinds_tensor = scatter_point_inds(indices, point_indices, output_shape)
    return v2pinds_tensor


