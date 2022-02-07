import torch
import torch.nn as nn
import torch.nn.functional as F
from . import voxel_query_utils
from typing import List


class NeighborVoxelSAModuleMSG(nn.Module):
                 
    def __init__(self, *, query_ranges: List[List[int]], radii: List[float], 
        nsamples: List[int], mlps: List[List[int]], use_xyz: bool = True, pool_method='max_pool'):
        """
        Args:
            query_ranges: list of int, list of neighbor ranges to group with
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(query_ranges) == len(nsamples) == len(mlps)
        
        self.groupers = nn.ModuleList()
        self.mlps_in = nn.ModuleList()
        self.mlps_pos = nn.ModuleList()
        self.mlps_out = nn.ModuleList()
        for i in range(len(query_ranges)):
            max_range = query_ranges[i] # [4, 4, 4]
            nsample = nsamples[i]  # [16]
            radius = radii[i] # [0.4]
            self.groupers.append(voxel_query_utils.VoxelQueryAndGrouping(max_range, radius, nsample))
            mlp_spec = mlps[i] # [32, 32, 32]

            cur_mlp_in = nn.Sequential(
                nn.Conv1d(mlp_spec[0], mlp_spec[1], kernel_size=1, bias=False), # (32, 32)
                nn.BatchNorm1d(mlp_spec[1])
            )
            
            cur_mlp_pos = nn.Sequential(
                nn.Conv2d(3, mlp_spec[1], kernel_size=1, bias=False), # (3, 32)
                nn.BatchNorm2d(mlp_spec[1])
            )

            cur_mlp_out = nn.Sequential(
                nn.Conv1d(mlp_spec[1], mlp_spec[2], kernel_size=1, bias=False), # (32, 32)
                nn.BatchNorm1d(mlp_spec[2]),
                nn.ReLU()
            )

            self.mlps_in.append(cur_mlp_in)
            self.mlps_pos.append(cur_mlp_pos)
            self.mlps_out.append(cur_mlp_out)

        self.relu = nn.ReLU()
        self.pool_method = pool_method # max_pool

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, \
                                        new_coords, features, voxel2point_indices):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features -->（204916, 3）特征点云坐标
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...] -->（8，）每帧点云有效坐标的个数
        :param new_xyz: (M1 + M2 ..., 4) -->（221184，4）roi grid点云坐标 221184 = 8 * 27648
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...] -->（8，）每帧roi grid有效坐标的个数 27648
        :param new_coords: (221184, 4)在该特征层上的roi voxle坐标
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features -->（204916, 32）有效点云特征
        :param point_indices: (B, Z, Y, X) tensor of point indices --> (8, 21, 800, 704) 有效voxle在原始空间shape中的索引
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        # change the order to [batch_idx, z, y, x]
        new_coords = new_coords[:, [0, 3, 2, 1]].contiguous() #（221184，4）roi grid点云坐标 221184 = 8 * 27648
        new_features_list = []
        # 逐个查询尺度处理，目前只有一个查询尺度
        for k in range(len(self.groupers)):
            # 1.对输入特征整体进行一次卷积，抽象特征，并调整维度，为group特征做准备
            # features_in: (1, C, M1+M2) 调整通道维度，并扩展batch维度
            features_in = features.permute(1, 0).unsqueeze(0) # (1, 32, 204916)
            # 对特征进行卷积
            features_in = self.mlps_in[k](features_in) # (1, 32, 204916)
            # features_in: (1, M1+M2, C)
            features_in = features_in.permute(0, 2, 1).contiguous() # (1, 204916，32)
            # features_in: (M1+M2, C)
            features_in = features_in.view(-1, features_in.shape[-1]) # (204916，32)

            # 2.以网格点为单位聚合其周围特征和坐标，获取论文中的M（每个网格点周围的特征和坐标）以下操作和论文中的图4是完全一致的
            # grouped_features: (M1+M2, C, nsample) --> (221184, 32, 16)
            # grouped_xyz: (M1+M2, 3, nsample) --> (221184, 3, 16)
            grouped_features, grouped_xyz, empty_ball_mask = self.groupers[k](
                new_coords, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features_in, voxel2point_indices
            )
            grouped_features[empty_ball_mask] = 0
            # 2.1特征提取 grouped_features: (1, C, M1+M2, nsample) --> (1, 32, 221184, 16)
            grouped_features = grouped_features.permute(1, 0, 2).unsqueeze(dim=0)
            # 2.2坐标残差 grouped_xyz: (M1+M2, 3, nsample)  --> (3, 221184, 16)
            grouped_xyz = grouped_xyz - new_xyz.unsqueeze(-1)
            grouped_xyz[empty_ball_mask] = 0
            # grouped_xyz: (1, 3, M1+M2, nsample) --> (1, 3, 221184, 16)
            grouped_xyz = grouped_xyz.permute(1, 0, 2).unsqueeze(0)
            # 2.3坐标残差升维 grouped_xyz: (1, C, M1+M2, nsample) --> (1, 32, 221184, 16)
            position_features = self.mlps_pos[k](grouped_xyz) 
            # 2.4特征相加 --> (1, 32, 221184, 16)
            new_features = grouped_features + position_features 
            new_features = self.relu(new_features)
            # 2.5最大池化
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, M1 + M2 ...) --> (1, 32, 221184) 在最后一维池化
            # 关于squeeze和unsqueeze的使用(只要调用pytorch的模块，输入维度和输出维度都是4维，默认要处理batch维度)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
            else:
                raise NotImplementedError
            
            # 3.对聚合特征进一步卷积抽象特征，并调整维度，加入features list
            new_features = self.mlps_out[k](new_features) # 在对新特征进行卷积 (1, 32, 221184)
            new_features = new_features.squeeze(dim=0).permute(1, 0)  # (M1 + M2 ..., C) --> (221184, 32)
            new_features_list.append(new_features)
        
        # (M1 + M2 ..., C)
        new_features = torch.cat(new_features_list, dim=1) # 拼接特征
        return new_features # (221184, 32)

