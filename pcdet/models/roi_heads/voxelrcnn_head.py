import torch
import torch.nn as nn
from ...ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate


class VoxelRCNNHead(RoIHeadTemplate):
    def __init__(self, backbone_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg # 模型配置
        self.pool_cfg = model_cfg.ROI_GRID_POOL # ROI_Head.ROI_GRID_POOL
        LAYER_cfg = self.pool_cfg.POOL_LAYERS # ROI_Head.ROI_GRID_POOL.POOL_LAYERS
        self.point_cloud_range = point_cloud_range # 点云范围
        self.voxel_size = voxel_size # voxel大小

        c_out = 0
        self.roi_grid_pool_layers = nn.ModuleList() # 初始化ROI网格池化层MuduleList
        # 逐池化采样层处理
        for src_name in self.pool_cfg.FEATURES_SOURCE: # FEATURES_SOURCE: ['x_conv2', 'x_conv3', 'x_conv4']
            mlps = LAYER_cfg[src_name].MLPS # 根据特征层获取MLP参数
            for k in range(len(mlps)): # MLPS: [[32, 32]] 长度为1
                # backbone_channels: {'x_conv1':16, 'x_conv2':32, 'x_conv3':64, 'x_conv4':64}
                mlps[k] = [backbone_channels[src_name]] + mlps[k] # 计算MLP层输入输出维度,在最前面增加一个值eg:[[32,32,32]]
            # 初始化池化层
            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                query_ranges=LAYER_cfg[src_name].QUERY_RANGES, # 查询范围
                nsamples=LAYER_cfg[src_name].NSAMPLE, # 采样数量
                radii=LAYER_cfg[src_name].POOL_RADIUS, # 池化半径 0.4->0.8->1.6
                mlps=mlps, # mlp层
                pool_method=LAYER_cfg[src_name].POOL_METHOD, # 池化方法
            )
            # 将池化层添加到ROI网格池化层MuduleList
            self.roi_grid_pool_layers.append(pool_layer)

            c_out += sum([x[-1] for x in mlps]) # 取mlps最后的输出维度 32->64->96


        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE # 获取网格大小 GRID_SIZE: 6
        # c_out = sum([x[-1] for x in mlps])
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out # 20736=6*6*6*96

        # 初始化共享FC层:SHARED_FC:[256, 256]
        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.SHARED_FC[k], bias=False), # 20736->256->256
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU(inplace=True)
            ])
            pre_channel = self.model_cfg.SHARED_FC[k] # 256

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0: # 如果不是最后一层，并且dropout比率>0
                shared_fc_list.aSppend(nn.Dropout(self.model_cfg.DP_RATIO)) # 添加dropout层，DP_RATIO: 0.3
        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        # 初始化分类层:CLS_FC:[256, 256]
        cls_fc_list = []
        for k in range(0, self.model_cfg.CLS_FC.__len__()):
            cls_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.CLS_FC[k], bias=False), # 256->256->256(在shared_fc_layer中pre_channel已经变为256)
                nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.CLS_FC[k]

            if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.cls_fc_layers = nn.Sequential(*cls_fc_list) 
        self.cls_pred_layer = nn.Linear(pre_channel, self.num_class, bias=True) # 分类层 256->1

        # 初始化回归层:REG_FC:[256, 256]
        reg_fc_list = []
        for k in range(0, self.model_cfg.REG_FC.__len__()):
            reg_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.REG_FC[k], bias=False), 
                nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.REG_FC[k]

            if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.reg_fc_layers = nn.Sequential(*reg_fc_list) # 256->256->256
        self.reg_pred_layer = nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True) # 回归层 256->7(7*1)

        self.init_weights()


    def init_weights(self):
        init_func = nn.init.xavier_normal_
        for module_list in [self.shared_fc_layer, self.cls_fc_layers, self.reg_fc_layers]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                    
        nn.init.normal_(self.cls_pred_layer.weight, 0, 0.01)
        nn.init.constant_(self.cls_pred_layer.bias, 0)
        nn.init.normal_(self.reg_pred_layer.weight, mean=0, std=0.001)
        nn.init.constant_(self.reg_pred_layer.bias, 0)
    

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        # ----------------------------------计算ROI坐标和统计点数---------------------------------- # 
        rois = batch_dict['rois'] # 获取采样后的ROI
        batch_size = batch_dict['batch_size'] # 获取batch size
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False) # False
        
        # 1.计算roi网格点全局点云坐标（旋转+roi中心点平移）
        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.pool_cfg.GRID_SIZE
        )  # (BxN, 6x6x6, 3) --> (1024, 216, 3)
        # roi_grid_xyz: (B, Nx6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)  # (8, 27648, 3)

        # 2.compute the voxel coordinates of grid points
        roi_grid_coords_x = (roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) // self.voxel_size[0]
        roi_grid_coords_y = (roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) // self.voxel_size[1]
        roi_grid_coords_z = (roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) // self.voxel_size[2]
        # roi_grid_coords: (B, Nx6x6x6, 3)
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1) # 整数坐标 --> (8, 27648, 3)

        # 3.逐帧赋值batch index
        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1) # (8, 27648, 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx 
        # roi_grid_coords: (B, Nx6x6x6, 4)
        # roi_grid_coords = torch.cat([batch_idx, roi_grid_coords], dim=-1)
        # roi_grid_coords = roi_grid_coords.int()
        
        # 4.计算每帧roi grid的有效坐标点数(虚拟特征点数)
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1]) # 8个27648

        # ------------------------------逐层提取ROI特征---------------------------------- # 
        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
            pool_layer = self.roi_grid_pool_layers[k] # 获取第k个池化层
            cur_stride = batch_dict['multi_scale_3d_strides'][src_name] # 获取该层下采样步长
            cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name] # 获取该层稀疏特征

            if with_vf_transform:
                cur_sp_tensors = batch_dict['multi_scale_3d_features_post'][src_name]
            else:
                cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

            # 1.compute voxel center xyz and batch_cnt
            cur_coords = cur_sp_tensors.indices # 提取有效voxle的坐标 --> (204916, 4)
            cur_voxel_xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4], # 第0维是batch index
                downsample_times=cur_stride, # 下采样倍数
                voxel_size=self.voxel_size, # voxle大小
                point_cloud_range=self.point_cloud_range # 点云范围
            ) # 有效voxle中心点云坐标 --> (204916, 3)
            
            # 2.统计每帧点云的有效坐标数
            cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
            
            # 3.get voxel2point tensor 计算空间voxle坐标与voxle特征之间的索引
            v2p_ind_tensor = common_utils.generate_voxel2pinds(cur_sp_tensors) # (8, 21, 800, 704)
            
            # 4.compute the grid coordinates in this scale, in [batch_idx, x y z] order
            cur_roi_grid_coords = roi_grid_coords // cur_stride # 计算下采样后的网格坐标 (8,27648,3)
            cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1) # 将batch index与roi grid coord拼接 --> (8,27648,4)
            cur_roi_grid_coords = cur_roi_grid_coords.int() # 转化为整数
            
            # 5.voxel neighbor aggregation
            pooled_features = pool_layer(
                xyz=cur_voxel_xyz.contiguous(), # voxle中心点云坐标
                xyz_batch_cnt=cur_voxel_xyz_batch_cnt, # 每帧点云有效坐标的个数
                new_xyz=roi_grid_xyz.contiguous().view(-1, 3), # roi grid点云坐标
                new_xyz_batch_cnt=roi_grid_batch_cnt, # 每个roi grid中有效坐标个数
                new_coords=cur_roi_grid_coords.contiguous().view(-1, 4), # 在该特征层上的roi voxle坐标
                features=cur_sp_tensors.features.contiguous(), # 稀疏特征
                voxel2point_indices=v2p_ind_tensor # 空间voxle坐标与voxle特征之间的索引(对应关系)
            )

            # 6.改变特征维度，并加入池化特征list
            pooled_features = pooled_features.view(
                -1, self.pool_cfg.GRID_SIZE ** 3,
                pooled_features.shape[-1]
            )  # (BxN, 6x6x6, C) --> (1024, 216, 32)
            pooled_features_list.append(pooled_features)
        
        ms_pooled_features = torch.cat(pooled_features_list, dim=-1)
        
        return ms_pooled_features # (1024, 216, 96)


    def get_global_grid_points_of_roi(self, rois, grid_size):
        """
        计算roi网格点全局点云坐标（旋转+roi中心点平移）
        Args:
            rois:(1024, 7)
            grid_size:6
        Returns:
            global_roi_grid_points, local_roi_grid_points: (1024, 216, 3)
        """
        rois = rois.view(-1, rois.shape[-1]) # (8, 128, 8) --> (1024, 8)
        batch_size_rcnn = rois.shape[0] # 1024

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3) --> (1024, 216, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1) # (1024, 216, 3) 前3维沿着z轴旋转
        global_center = rois[:, 0:3].clone() # 提取roi的中心坐标 (1024,3)
        global_roi_grid_points += global_center.unsqueeze(dim=1) # 将box平移到roi的中心 (1024, 216, 3)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        """
        根据roi的长宽高计算稠密的虚拟点云坐标(roi box划分为6x6x6的网格坐标)
        Args:
            rois:(1024, 7)
            batch_size_rcnn:1024
            grid_size:6
        Returns:
            roi_grid_points: (1024, 216, 3)
        """
        faked_features = rois.new_ones((grid_size, grid_size, grid_size)) # 初始化一个全1的6x6x6的伪特征
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx] --> (216,3)
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3) --> (1024, 216, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6] # 取出roi的长宽高(1024,3)
        # ROI网格点坐标：先平移0.5个单位，然后归一化，再乘roi的大小，最后将原点移动中心
        # (1024,216,3) / (1024,1,3) - (1024,1,3)
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3) 
        return roi_grid_points # (1024, 216, 3)


    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        # 1.根据预测结果，进行NMS后生成512个ROI(proposal)
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        # 2.为ROI分配GT同时根据iou阈值区分前景和背景并采样128个prorosal
        if self.training:
            """
            targets_dict = {
                'rois': 采样后的ROI (8,128,7)
                'gt_of_rois': 采样后ROI对应的gt(编码修正后) (8,128,8)
                'gt_iou_of_rois': 采样后ROI与gt的最大iou (8,128)
                'roi_scores': 采样后ROI分数 (8,128）
                'roi_labels': 采样后ROI标签 (8,128)
                'reg_valid_mask': 回归mask (8, 128)
                'rcnn_cls_labels': 类别标签 (8, 128)
                } 
            """
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # 3.RoI aware pooling ROI grid池化:根据坐标在不同特征层上提取并聚合proposal特征
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C) --> (1024, 216, 96)

        # 4.Box Refinement
        pooled_features = pooled_features.view(pooled_features.size(0), -1) # (1024, 20736)
        shared_features = self.shared_fc_layer(pooled_features) # (1024, 256)
        rcnn_cls = self.cls_pred_layer(self.cls_fc_layers(shared_features)) # (1024, 1)
        rcnn_reg = self.reg_pred_layer(self.reg_fc_layers(shared_features)) # (1024, 7)

        # grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        # batch_size_rcnn = pooled_features.shape[0]
        # pooled_features = pooled_features.permute(0, 2, 1).\
        #     contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        # shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        # rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        # rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls # 将预测结果写入target_dict
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict
