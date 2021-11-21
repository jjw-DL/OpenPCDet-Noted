import torch

from .vfe_template import VFETemplate


class MeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features # 4

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C) (64000, 5, 4)
                voxel_num_points: optional (num_voxels) (64000)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points'] # (64000, 5, 4), (64000)
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False) # 求每个voxel内点坐标的和 # (64000, 4)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features) # 正则化-->(64000, 1), 防止除0
        points_mean = points_mean / normalizer # 求每个voxel内点坐标的平均值 # (64000, 4)
        batch_dict['voxel_features'] = points_mean.contiguous() # 将voxel_features信息加入batch_dict

        return batch_dict
