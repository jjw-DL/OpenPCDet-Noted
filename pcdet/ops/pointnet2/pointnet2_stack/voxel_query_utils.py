import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
from typing import List

from . import pointnet2_stack_cuda as pointnet2
from . import pointnet2_utils

class VoxelQuery(Function):

    @staticmethod
    def forward(ctx, max_range: int, radius: float, nsample: int, xyz: torch.Tensor, \
                    new_xyz: torch.Tensor, new_coords: torch.Tensor, point_indices: torch.Tensor):
        """
        Args:
            ctx:
            max_range: int, max range of voxels to be grouped （voxle的单位）
            radius:点云坐标半径范围
            nsample: int, maximum number of features in the balls
            xyz:voxel点云坐标 (204916, 3)
            new_xyz: roi grid点云坐标 (221184, 3) --> 两组点云坐标用于计算距离，配合raduis判断点是否合格
            new_coords: (M1 + M2, 4), [batch_id, z, y, x] cooridnates of keypoints 用于查询
            point_indices: (batch_size, Z, Y, X) 4-D tensor recording the point indices of voxels
        Returns:
            idx: (M1 + M2, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()
        assert new_coords.is_contiguous()
        assert point_indices.is_contiguous()

        M = new_coords.shape[0] # 221184
        B, Z, Y, X = point_indices.shape # 8, 21, 800, 704
        idx = torch.cuda.IntTensor(M, nsample).zero_() # (221184, 16)

        z_range, y_range, x_range = max_range # 4, 4, 4
        pointnet2.voxel_query_wrapper(M, Z, Y, X, nsample, radius, z_range, y_range, x_range, \
                    new_xyz, xyz, new_coords, point_indices, idx)

        empty_ball_mask = (idx[:, 0] == -1) # 将第一个neighbor的index为-1的网格点标记为未查询到neighbor
        idx[empty_ball_mask] = 0 # 将未查询到neighbor的网格点的neighbor index设置为0

        return idx, empty_ball_mask

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None

voxel_query = VoxelQuery.apply


class VoxelQueryAndGrouping(nn.Module):
    def __init__(self, max_range: int, radius: float, nsample: int):
        """
        Args:
            radius: float, radius of ball
            nsample: int, maximum number of features to gather in the ball
        """
        super().__init__()
        self.max_range, self.radius, self.nsample = max_range, radius, nsample # [4,4,4], [0.4], [16]

    def forward(self, new_coords: torch.Tensor, xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor,
                new_xyz: torch.Tensor, new_xyz_batch_cnt: torch.Tensor,
                features: torch.Tensor, voxel2point_indices: torch.Tensor):
        """
        Args:
            new_coords: (M1 + M2 ..., 3) centers voxel indices of the ball query
            xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...] 点云中有效坐标的数量
            new_xyz: (M1 + M2 ..., 3) centers of the ball query
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...] # roi grid中有效坐标的数量
            features: (N1 + N2 ..., C) tensor of features to group
            voxel2point_indices: (B, Z, Y, X) tensor of points indices of voxels

        Returns:
            new_features: (M1 + M2, C, nsample) tensor
        """
        assert xyz.shape[0] == xyz_batch_cnt.sum(), 'xyz: %s, xyz_batch_cnt: %s' % (str(xyz.shape), str(new_xyz_batch_cnt))
        assert new_coords.shape[0] == new_xyz_batch_cnt.sum(), \
            'new_coords: %s, new_xyz_batch_cnt: %s' % (str(new_coords.shape), str(new_xyz_batch_cnt))
        batch_size = xyz_batch_cnt.shape[0] # 获取batch size:8
        
        # idx: (M1 + M2 ..., nsample) --> (221184, 16) roi网格点的聚合特征位置索引
        # empty_ball_mask: (M1 + M2 ...) --> (221184,) roi网格点是否成功聚合特征
        # max_range:[4,4,4], radius:[0.4], nsample:[16]
        idx1, empty_ball_mask1 = voxel_query(self.max_range, self.radius, self.nsample, xyz, new_xyz, new_coords, voxel2point_indices)

        idx1 = idx1.view(batch_size, -1, self.nsample) # (8, 27648, 16)
        count = 0
        # 按照batch减去对应起始地址，计算帧内相对索引
        for bs_idx in range(batch_size):
            idx1[bs_idx] -= count
            count += xyz_batch_cnt[bs_idx]
        idx1 = idx1.view(-1, self.nsample) # (221184, 16)
        idx1[empty_ball_mask1] = 0 # 将未查询到点的位置赋0

        idx = idx1 # (221184, 16) 
        empty_ball_mask = empty_ball_mask1 # (221184,)
        """
        xyz:特征点云坐标 (204916, 3)
        xyz_batch_cnt:每帧点云的特征点数量 (8,)
        idx:网格点neighbor的index (221184, 16)
        new_xyz_batch_cnt:每帧点云的roi网格点数量(8,)
        """
        grouped_xyz = pointnet2_utils.grouping_operation(xyz, xyz_batch_cnt, idx, new_xyz_batch_cnt) # (221184, 3, 16)
        # grouped_features: (M1 + M2, C, nsample)
        grouped_features = pointnet2_utils.grouping_operation(features, xyz_batch_cnt, idx, new_xyz_batch_cnt) # (221184, 32, 16)
        
        return grouped_features, grouped_xyz, empty_ball_mask
