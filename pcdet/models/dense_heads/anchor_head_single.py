import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate


class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        """
        Args:
            model_cfg: AnchorHeadSingle的配置
            input_channels:384 输入通道数
            num_class: 3
            class_names: ['Car','Pedestrian','Cyclist']
            grid_size: (432,493,1)
            point_cloud_range:(0, -39.68, -3, 69.12, 39.68, 1)
            predict_boxes_when_training:False
        """
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location) # 2*3=6
        # Conv2d(384,18,kernel_size=(1,1),stride=(1,1))
        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class, # 6*3=18
            kernel_size=1
        )
        # Conv2d(384,42,kernel_size=(1,1),stride=(1,1))
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size, # 6*7=42
            kernel_size=1
        )
        # 如果存在方向损失，则添加方向卷积层Conv2d(384,12,kernel_size=(1,1),stride=(1,1))
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS, # 6*2=12
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        # 参数初始化
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d'] # （4，384，248，216）

        cls_preds = self.conv_cls(spatial_features_2d) # 每个anchor的类别预测-->(4,18,248,216)
        box_preds = self.conv_box(spatial_features_2d) # 每个anchor的box预测-->(4,42,248,216)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C] -->(4,248,216,42)
        # 将预测结果存入前传结果字典
        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        # 如果存在方向卷积层，则继续处理方向
        if self.conv_dir_cls is not None: 
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d) # 每个anchor的方向预测-->(4,12,248,216)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C] -->(4,248,216,12)
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            # targets_dict = {
            #     'box_cls_labels': cls_labels, # (4，321408）
            #     'box_reg_targets': bbox_targets, # (4，321408，7）
            #     'reg_weights': reg_weights # (4，321408）
            # }
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes'] # （4，39，8）
            )
            self.forward_ret_dict.update(targets_dict)

        # 如果不是训练模式，则直接进行box的预测或对于双阶段网络要生成proposal(此时batch不为1)
        if not self.training or self.predict_boxes_when_training:
            # 输入为最开始的类别和box以及方向的预测，输出为展开后的预测
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            ) 
            data_dict['batch_cls_preds'] = batch_cls_preds # (1, 321408, 3)
            data_dict['batch_box_preds'] = batch_box_preds # (1, 321408, 7)
            data_dict['cls_preds_normalized'] = False

        return data_dict
