import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import box_coder_utils, common_utils, loss_utils
from ..model_utils.model_nms_utils import class_agnostic_nms
from .target_assigner.proposal_target_layer import ProposalTargetLayer


class RoIHeadTemplate(nn.Module):
    def __init__(self, num_class, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg # ROI_HEAD模型配置
        self.num_class = num_class # 类别数量:1 默认值
        self.box_coder = getattr(box_coder_utils, self.model_cfg.TARGET_CONFIG.BOX_CODER)(
            **self.model_cfg.TARGET_CONFIG.get('BOX_CODER_CONFIG', {}) # ResidualCoder 传入的是空字典，默认参数即可
        )
        self.proposal_target_layer = ProposalTargetLayer(roi_sampler_cfg=self.model_cfg.TARGET_CONFIG)
        self.build_losses(self.model_cfg.LOSS_CONFIG) # 构建损失: BinaryCrossEntropy和smooth-l1
        self.forward_ret_dict = None

    def build_losses(self, losses_cfg):
        self.add_module(
            'reg_loss_func',
            loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights']) 
            # 'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        ) # 在ROIHeadTemplate模块中添加子模块(pytorch原始函数) --> self._modules[name] = module

    def make_fc_layers(self, input_channels, output_channels, fc_list):
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            if self.model_cfg.DP_RATIO >= 0 and k == 0:
                fc_layers.append(nn.Dropout(self.model_cfg.DP_RATIO))
        fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    @torch.no_grad()
    def proposal_layer(self, batch_dict, nms_config):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized # False
                batch_index: optional (N1+N2+...)
            nms_config:
                NMS_TYPE: nms_gpu
                MULTI_CLASSES_NMS: False
                NMS_PRE_MAXSIZE: 9000
                NMS_POST_MAXSIZE: 512
                NMS_THRESH: 0.8

        Returns:
            batch_dict:
                rois: (B, num_rois, 7+C)
                roi_scores: (B, num_rois)
                roi_labels: (B, num_rois)
        """
        # 1.获取所需数据
        batch_size = batch_dict['batch_size'] # 8
        batch_box_preds = batch_dict['batch_box_preds'] # （8, 70400, 7）
        batch_cls_preds = batch_dict['batch_cls_preds'] # （8, 70400, 1）
        # 2.初始化roi相关要素(返回值)
        rois = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE, batch_box_preds.shape[-1])) # (8, 512,7)
        roi_scores = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE)) # (8, 512)
        roi_labels = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE), dtype=torch.long) # (8, 512)

        # 3.逐帧处理
        for index in range(batch_size):
            # 3.1 计算batch_mask,提取当前帧box和cls预测值
            if batch_dict.get('batch_index', None) is not None:
                assert batch_cls_preds.shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index
            box_preds = batch_box_preds[batch_mask] # （70400, 7）
            cls_preds = batch_cls_preds[batch_mask] # （70400, 1）
            
            # 3.2 计算roi的分数和标签
            cur_roi_scores, cur_roi_labels = torch.max(cls_preds, dim=1) # (70400)

            if nms_config.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                # 3.3 进行NMS得到k个被保留的box的索引和分数
                selected, selected_scores = class_agnostic_nms(
                    box_scores=cur_roi_scores, box_preds=box_preds, nms_config=nms_config
                )
            # 3.4 将被保留的box，scores以及对应的labels赋予roi
            rois[index, :len(selected), :] = box_preds[selected]
            roi_scores[index, :len(selected)] = cur_roi_scores[selected]
            roi_labels[index, :len(selected)] = cur_roi_labels[selected]

        batch_dict['rois'] = rois # (8, 512,7)
        batch_dict['roi_scores'] = roi_scores # (8, 512)
        batch_dict['roi_labels'] = roi_labels + 1 # 类别索引+1 # (8, 512)
        batch_dict['has_class_labels'] = True if batch_cls_preds.shape[-1] > 1 else False
        batch_dict.pop('batch_index', None) # 删除字典值'batch_index',若不存在则返回默认值None
        return batch_dict

    def assign_targets(self, batch_dict):
        batch_size = batch_dict['batch_size'] # 获取batch size
        with torch.no_grad():
            """
            targets_dict = {
                'rois': 采样后的ROI (8,128,7)
                'gt_of_rois': 采样后ROI对应的gt (8,128,8)
                'gt_iou_of_rois': 采样后ROI与gt的最大iou (8,128)
                'roi_scores': 采样后ROI分数 (8,128）
                'roi_labels': 采样后ROI标签 (8,128)
                'reg_valid_mask': 回归mask (8, 128)
                'rcnn_cls_labels': 类别标签 (8, 128)
                } 
            """
            targets_dict = self.proposal_target_layer.forward(batch_dict)
        # 以下部分都是在编码gt_of_rois
        rois = targets_dict['rois']  # (B, N, 7 + C) (8,128,7)
        gt_of_rois = targets_dict['gt_of_rois']  # (B, N, 7 + C + 1) (8,128,8) 最后一维是类别
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach() # 复制roi对应gt

        # 1.canonical transformation
        roi_center = rois[:, :, 0:3] # 提取roi中心
        roi_ry = rois[:, :, 6] % (2 * np.pi) # 提取roi旋转角
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center # 计算gt与roi中心的差
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry # 计算gt与roi旋转角的差

        # 2.transfer LiDAR coords to local coords
        gt_of_rois = common_utils.rotate_points_along_z(
            points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)
        ).view(batch_size, -1, gt_of_rois.shape[-1]) # (8,128,8)-->(1024,1,8)-->(8,128,8)

        # 3.flip orientation if rois have opposite orientation
        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
        heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2) # 将heading_label截取到-pi/2到pi/2之间

        gt_of_rois[:, :, 6] = heading_label
        targets_dict['gt_of_rois'] = gt_of_rois # 修正的gt_of_rois (8, 128, 8)
        return targets_dict

    def get_box_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG # REG_LOSS: smooth-l1
        code_size = self.box_coder.code_size # 7
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1) # 有效box的mask (512,)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size] # (4, 128, 7) gt值
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size) # (512, 7)
        rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C) # (512, 7) 预测值
        roi_boxes3d = forward_ret_dict['rois'] # (4, 128, 7) proposal值(anchor)
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0] # 512

        fg_mask = (reg_valid_mask > 0) # (512,)
        fg_sum = fg_mask.long().sum().item() # 2

        tb_dict = {}

        if loss_cfgs.REG_LOSS == 'smooth-l1':
            # 将roi作为anchor的原因是后面的的预测值是在roi预测值的基础上的微调
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0 # 将anchor的xyz设置为0
            rois_anchor[:, 6] = 0 # 将anchor的旋转角设置为0
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
            ) # 对gt进行编码(512, 7)

            rcnn_loss_reg = self.reg_loss_func(
                rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0), # 预测值(512, 7)
                reg_targets.unsqueeze(dim=0), # (512, 7)
            )  # [B, M, 7] --> (1, 512, 7)
            # 针对前景进行mask
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight'] # 乘回归权重
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item() # 记录损失

            if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:
                # TODO: NEED to BE CHECK
                fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask] # (2, 7) 预测值
                fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask] # (2, 7) anchor

                fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                batch_anchors = fg_roi_boxes3d.clone().detach()
                roi_ry = fg_roi_boxes3d[:, :, 6].view(-1) # 取出旋转角 --> (2, 1)
                roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3) # 取出xyz --> (2, 3)
                batch_anchors[:, :, 0:3] = 0 # 将anchor的xyz设置为0
                rcnn_boxes3d = self.box_coder.decode_torch(
                    fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
                ).view(-1, code_size) # 根据anchor对预测box进行恢复 # (2, 7)

                rcnn_boxes3d = common_utils.rotate_points_along_z(
                    rcnn_boxes3d.unsqueeze(dim=1), roi_ry
                ).squeeze(dim=1) # 将box旋转ry角度
                rcnn_boxes3d[:, 0:3] += roi_xyz # 加上roi的中心

                loss_corner = loss_utils.get_corner_loss_lidar(
                    rcnn_boxes3d[:, 0:7],
                    gt_of_rois_src[fg_mask][:, 0:7]
                ) # 计算角点损失
                loss_corner = loss_corner.mean() # 求平均
                loss_corner = loss_corner * loss_cfgs.LOSS_WEIGHTS['rcnn_corner_weight'] # 乘权重

                rcnn_loss_reg += loss_corner # 加损失
                tb_dict['rcnn_loss_corner'] = loss_corner.item() # 记录损失
        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict

    def get_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG # CLS_LOSS: BinaryCrossEntropy
        rcnn_cls = forward_ret_dict['rcnn_cls'] # (512, 1)
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1) # (512,) 在assign时分配gt
        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1) # (512,)
            # 对预测值进行sigmoid，调用二分类交叉熵损失
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float() # 只计算care的物体的分类
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight'] # 乘权重
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()} # 记录损失
        return rcnn_loss_cls, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        # 计算分类损失
        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls # 将分类损失加入总损失
        tb_dict.update(cls_tb_dict) # 更新tensorboard分类损失

        # 计算回归损失
        rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_reg # 将回归损失加入总损失
        tb_dict.update(reg_tb_dict) # 更新tensorboard回归损失
        tb_dict['rcnn_loss'] = rcnn_loss.item() # 将总损失加入tensorboard损失
        return rcnn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)

        Returns:

        """
        code_size = self.box_coder.code_size
        # batch_cls_preds: (B, N, num_class or 1)
        batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1])
        batch_box_preds = box_preds.view(batch_size, -1, code_size)

        roi_ry = rois[:, :, 6].view(-1)
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0

        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)

        batch_box_preds = common_utils.rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)
        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)
        return batch_cls_preds, batch_box_preds
