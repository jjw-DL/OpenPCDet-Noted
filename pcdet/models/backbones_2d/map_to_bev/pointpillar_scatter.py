import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    """
    对应到论文中就是stacked pillars，将生成的pillar按照坐标索引还原到原空间中
    """
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES # 64
        self.nx, self.ny, self.nz = grid_size  # [432,496,1]
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            pillar_features:(31530,64)
            coords:(31530, 4) 第一维是batch_index
        Returns:
            batch_spatial_features:(4, 64, 496, 432)
        """
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1 # 4
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device) # (64,214272)-->1x432x496=214272
            # batch_index的mask
            batch_mask = coords[:, 0] == batch_idx
            # 根据mask提取坐标
            this_coords = coords[batch_mask, :] # (8629,4) 
            # 这里的坐标是z,y和x的形式,且只有一层，因此计算索引的方式如下
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            # 转换数据类型
            indices = indices.type(torch.long)
            # 根据mask提取pillar_features
            pillars = pillar_features[batch_mask, :] # (8629,64)
            pillars = pillars.t() # (64,8629)
            # 在索引位置填充pillars
            spatial_feature[:, indices] = pillars
            # 将空间特征加入list,每个元素为(64,214272)
            batch_spatial_features.append(spatial_feature) 

        batch_spatial_features = torch.stack(batch_spatial_features, 0) # (4, 64, 214272)
        # reshape回原空间(伪图像)--> (4, 64, 496, 432)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        # 将结果加入batch_dict
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict
