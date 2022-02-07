import numpy as np
import torch

from ....ops.iou3d_nms import iou3d_nms_utils
from ....utils import box_utils


class AxisAlignedTargetAssigner(object):
    def __init__(self, model_cfg, class_names, box_coder, match_height=False):
        super().__init__()

        anchor_generator_cfg = model_cfg.ANCHOR_GENERATOR_CONFIG # anchor生成配置参数
        anchor_target_cfg = model_cfg.TARGET_ASSIGNER_CONFIG # 为预测box找对应anchor的参数
        self.box_coder = box_coder # pcdet.utils.box_coder_utils.ResidualCoder
        self.match_height = match_height # False
        self.class_names = np.array(class_names) # ['Car', 'Pedestrian', 'Cyclist']
        self.anchor_class_names = [config['class_name'] for config in anchor_generator_cfg] # ['Car', 'Pedestrian', 'Cyclist']
        self.pos_fraction = anchor_target_cfg.POS_FRACTION if anchor_target_cfg.POS_FRACTION >= 0 else None  # anchor_target_cfg.POS_FRACTION = -1 < 0 --> None
        self.sample_size = anchor_target_cfg.SAMPLE_SIZE # 512
        self.norm_by_num_examples = anchor_target_cfg.NORM_BY_NUM_EXAMPLES # False
        self.matched_thresholds = {} # {'Car':0.6, 'Pedestrian':0.5, 'Cyclist':0.5}
        self.unmatched_thresholds = {} # {'Car':0.45, 'Pedestrian':0.35, 'Cyclist':0.35}
        for config in anchor_generator_cfg:
            self.matched_thresholds[config['class_name']] = config['matched_threshold']
            self.unmatched_thresholds[config['class_name']] = config['unmatched_threshold']

        self.use_multihead = model_cfg.get('USE_MULTIHEAD', False) # False
        # self.separate_multihead = model_cfg.get('SEPARATE_MULTIHEAD', False)
        # if self.seperate_multihead:
        #     rpn_head_cfgs = model_cfg.RPN_HEAD_CFGS
        #     self.gt_remapping = {}
        #     for rpn_head_cfg in rpn_head_cfgs:
        #         for idx, name in enumerate(rpn_head_cfg['HEAD_CLS_NAME']):
        #             self.gt_remapping[name] = idx + 1

    def assign_targets(self, all_anchors, gt_boxes_with_classes):
        """
        处理一个batch中所有点云的anchors和gt_boxes，计算前景和背景anchor的类别，box编码和回归权重
        Args:
            all_anchors: [(N, 7), ...] [(1，248，216，1，2，7），(1，248，216，1，2，7），(1，248，216，1，2，7)]
            gt_boxes: (B, M, 8)
        Returns:
            all_targets_dict = {
                'box_cls_labels': cls_labels, # (4，321408）
                'box_reg_targets': bbox_targets, # (4，321408，7）
                'reg_weights': reg_weights # (4，321408）
            }
        """
        # 1.初始化结果list并提取对应的gt_box和类别
        bbox_targets = []
        cls_labels = []
        reg_weights = []

        batch_size = gt_boxes_with_classes.shape[0] # 4
        gt_classes = gt_boxes_with_classes[:, :, -1] #（4，39）
        gt_boxes = gt_boxes_with_classes[:, :, :-1] #（4，39，7）

        # 2.按照batch逐帧计算anchor的前景和背景
        for k in range(batch_size):
            cur_gt = gt_boxes[k] # 取出当前gt_boxes (39，7）
            cnt = cur_gt.__len__() - 1 # 38
            # 这里的循环是找到最后一个非零的box，因为预处理的时候会按照batch最大box的数量处理，不足的进行补0
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            # 2.1提取当前帧非零的box和类别
            cur_gt = cur_gt[:cnt + 1]
            cur_gt_classes = gt_classes[k][:cnt + 1].int()

            target_list = []
            # 2.2 按照类别和anchors计算anchor的前景和背景
            for anchor_class_name, anchors in zip(self.anchor_class_names, all_anchors):
                # anchors:(1,248,216,1,2,7)
                if cur_gt_classes.shape[0] > 1:
                    # 这里减1是因为索引从0开始，目的是找到当前类别的mask
                    mask = torch.from_numpy(self.class_names[cur_gt_classes.cpu() - 1] == anchor_class_name)
                else:
                    mask = torch.tensor([self.class_names[c - 1] == anchor_class_name
                                         for c in cur_gt_classes], dtype=torch.bool)

                if self.use_multihead: # False
                    anchors = anchors.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchors.shape[-1])
                    # if self.seperate_multihead:
                    #     selected_classes = cur_gt_classes[mask].clone()
                    #     if len(selected_classes) > 0:
                    #         new_cls_id = self.gt_remapping[anchor_class_name]
                    #         selected_classes[:] = new_cls_id
                    # else:
                    #     selected_classes = cur_gt_classes[mask]
                    selected_classes = cur_gt_classes[mask]
                else:
                    # 2.2.1 计算所需的变量
                    feature_map_size = anchors.shape[:3] #（1，248，216）
                    anchors = anchors.view(-1, anchors.shape[-1]) # (107136,7) 107136=1x248x216x1x2
                    selected_classes = cur_gt_classes[mask] # 被选择的类别 （15，）
                # 2.2.2 调用assign_targets_single计算某一类别的anchors和gt_boxes，计算前景和背景anchor的类别，box编码和回归权重
                single_target = self.assign_targets_single(
                    anchors,
                    cur_gt[mask],
                    gt_classes=selected_classes,
                    matched_threshold=self.matched_thresholds[anchor_class_name], # 根据类别选择matched_thresholds和unmatched_thresholds
                    unmatched_threshold=self.unmatched_thresholds[anchor_class_name]  
                )
                target_list.append(single_target)
            # 到目前为止，处理完该帧所有类别和anchor的前景和背景

            if self.use_multihead:
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(-1) for t in target_list],
                    'box_reg_targets': [t['box_reg_targets'].view(-1, self.box_coder.code_size) for t in target_list],
                    'reg_weights': [t['reg_weights'].view(-1) for t in target_list]
                }

                target_dict['box_reg_targets'] = torch.cat(target_dict['box_reg_targets'], dim=0)
                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=0).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=0).view(-1)
            else:
                # 对该帧各类anchor的assign结果进行view并拼接
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(*feature_map_size, -1) for t in target_list], # feature_map_size:(1，248，216）--> (1，248，216, 2)
                    'box_reg_targets': [t['box_reg_targets'].view(*feature_map_size, -1, self.box_coder.code_size) # (1，248，216, 2, 7)
                                        for t in target_list],
                    'reg_weights': [t['reg_weights'].view(*feature_map_size, -1) for t in target_list] # (1，248，216, 2)
                }
                target_dict['box_reg_targets'] = torch.cat(
                    target_dict['box_reg_targets'], dim=-2
                ).view(-1, self.box_coder.code_size) # list:3 (1，248，216, 2, 7) --> (1，248，216, 6, 7) -> (321408, 7)

                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=-1).view(-1) # list:3 (1，248，216, 2) --> (1，248，216, 6) -> (321408,)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=-1).view(-1) # list:3 (1，248，216, 2) --> (1，248，216, 6) -> (321408,)

            # 将结果填入对应的容器
            bbox_targets.append(target_dict['box_reg_targets'])
            cls_labels.append(target_dict['box_cls_labels'])
            reg_weights.append(target_dict['reg_weights'])
        # 到这里该batch的点云全部处理完

        # 3.将结果stack并返回
        bbox_targets = torch.stack(bbox_targets, dim=0) # (4，321408，7）

        cls_labels = torch.stack(cls_labels, dim=0) # (4，321408）
        reg_weights = torch.stack(reg_weights, dim=0) # (4，321408）
        all_targets_dict = {
            'box_cls_labels': cls_labels, # (4，321408）
            'box_reg_targets': bbox_targets, # (4，321408，7）
            'reg_weights': reg_weights # (4，321408）
        }
        return all_targets_dict

    def assign_targets_single(self, anchors, gt_boxes, gt_classes, matched_threshold=0.6, unmatched_threshold=0.45):
        """
        针对某一类别的anchors和gt_boxes，计算前景和背景anchor的类别，box编码和回归权重
        Args:
            anchors: (107136,7)
            gt_boxes: （15，7）
            gt_classes: (15,1)
            matched_threshold:0.6
            unmatched_threshold:0.45
        Returns:
        前景anchor
            ret_dict = {
                'box_cls_labels': labels, # (107136,)
                'box_reg_targets': bbox_targets,  # (107136,7)
                'reg_weights': reg_weights, # (107136,)
            }
        """
        #----------------------------1.初始化-------------------------------#
        num_anchors = anchors.shape[0] # 107136
        num_gt = gt_boxes.shape[0] # 15

        # 初始化anchor对应的label和gt_id
        labels = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1
        gt_ids = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1

        #---------------------2.计算anchor的前景和背景------------------------#
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            # 1.计算gt和anchors之间的overlap
            anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(anchors[:, 0:7], gt_boxes[:, 0:7]) \
                if self.match_height else box_utils.boxes3d_nearest_bev_iou(anchors[:, 0:7], gt_boxes[:, 0:7]) # （107136，15）
                
            # 2.找到每个anchor最匹配的gt的索引和iou
            # anchor_to_gt_argmax表示数据维度是anchor的长度，索引是gt
            anchor_to_gt_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=1)).cuda() # (107136，）找到每个anchor最匹配的gt的索引
            anchor_to_gt_max = anchor_by_gt_overlap[
                torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax
            ] # （107136，）找到每个anchor最匹配的gt的iou

            # 3.找到每个gt最匹配anchor的索引和iou
            gt_to_anchor_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=0)).cuda() # (15,) 找到每个gt最匹配anchor的索引
            gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, torch.arange(num_gt, device=anchors.device)] # （15，）找到每个gt最匹配anchor的iou
            # 4.标记没有匹配的gt并将iou置为-1
            empty_gt_mask = gt_to_anchor_max == 0 # 没有匹配anchor的gt的mask
            gt_to_anchor_max[empty_gt_mask] = -1 # 让没有匹配anchor的gt的iou值为-1

            # 5.找到anchor中和gt存在最大iou的anchor索引，即前景anchor
            # 以gt为基础，逐个anchor对应，比如第一个gt的最大iou为0.9，则在所有anchor中找iou为0.9的anchor
            anchors_with_max_overlap = (anchor_by_gt_overlap == gt_to_anchor_max).nonzero()[:, 0] # (35,)
            # 找到anchor中和gt存在最大iou的gt索引
            # 其实和(anchor_by_gt_overlap == gt_to_anchor_max).nonzero()[:, 1]的结果一样
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap] # （35，）

            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force] # 将gt的类别赋值到对应的anchor的label中
            gt_ids[anchors_with_max_overlap] = gt_inds_force.int() # 将gt的索引赋值到对应的anchor的gt_id中

            # 6.根据matched_threshold和unmatched_threshold以及anchor_to_gt_max计算前景和背景索引，并更新labels和gt_ids
            # 这里应该对labels和gt_ids的操作应该包含了上面的anchors_with_max_overlap
            pos_inds = anchor_to_gt_max >= matched_threshold # 找到最匹配的anchor中iou大于给定阈值的mask #(107136,)
            gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds] # 找到最匹配的anchor中iou大于给定阈值的gt的索引 #(105,)
            labels[pos_inds] = gt_classes[gt_inds_over_thresh] # 将pos anchor对应gt的类别赋值到对应的anchor的label中
            gt_ids[pos_inds] = gt_inds_over_thresh.int() # 将pos anchor对应gt的索引赋值到对应的anchor的gt_id中

            bg_inds = (anchor_to_gt_max < unmatched_threshold).nonzero()[:, 0] # 找到背景anchor索引 (106879，)
        else:
            bg_inds = torch.arange(num_anchors, device=anchors.device)

        fg_inds = (labels > 0).nonzero()[:, 0] # 找到前景anchor的索引--> (119,)  106879 + 119 = 106998 < 107136 有一些anchor既不是背景也不是前景
        # 到目前为止得到了anchor的前景和背景

        #------------------3.对anchor的前景和背景进行筛选和赋值--------------------#
        # 如果存在前景采样比例，则分别采样前景和背景anchor 
        if self.pos_fraction is not None: # anchor_target_cfg.POS_FRACTION = -1 < 0 --> None
            num_fg = int(self.pos_fraction * self.sample_size) # self.sample_size=512
            # 如果前景anchor大于采样前景数
            if len(fg_inds) > num_fg:
                # 计算要丢弃的前景anchor数目
                num_disabled = len(fg_inds) - num_fg
                # 在前景数目中随机产生索引值，并取前num_disabled个关闭索引
                # 比如：torch.randperm(4)
                # 输出：tensor([ 2,  1,  0,  3])
                disable_inds = torch.randperm(len(fg_inds))[:num_disabled]
                # 将被丢弃的anchor的iou设置为-1
                labels[disable_inds] = -1
                # 更新前景索引
                fg_inds = (labels > 0).nonzero()[:, 0]
            
            # 计算所需背景数
            num_bg = self.sample_size - (labels > 0).sum()
            # 如果当前背景数大于所需背景数
            if len(bg_inds) > num_bg:
                # torch.randint在0到len(bg_inds)之间，随机产生size为(num_bg,)的数组
                enable_inds = bg_inds[torch.randint(0, len(bg_inds), size=(num_bg,))]
                # 将enable_inds的标签设置为0
                labels[enable_inds] = 0
            # bg_inds = torch.nonzero(labels == 0)[:, 0]
        else:
            if len(gt_boxes) == 0 or anchors.shape[0] == 0:
                labels[:] = 0
            else:
                # 将背景赋0
                labels[bg_inds] = 0
                # 将前景赋对应类别
                labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]

        #------------------4.计算bbox_targets和reg_weights--------------------#
        # 初始化bbox_targets
        bbox_targets = anchors.new_zeros((num_anchors, self.box_coder.code_size)) # (107136,7)
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            fg_gt_boxes = gt_boxes[anchor_to_gt_argmax[fg_inds], :] # 提取前景对应的gt box
            fg_anchors = anchors[fg_inds, :] # 提取前景anchor
            bbox_targets[fg_inds, :] = self.box_coder.encode_torch(fg_gt_boxes, fg_anchors) # 编码gt和前景anchor，并赋值到bbox_targets的对应位置

        # 初始化回归权重
        reg_weights = anchors.new_zeros((num_anchors,)) # (107136,)

        if self.norm_by_num_examples: # False
            num_examples = (labels >= 0).sum()
            num_examples = num_examples if num_examples > 1.0 else 1.0
            reg_weights[labels > 0] = 1.0 / num_examples
        else:
            reg_weights[labels > 0] = 1.0 # 将前景anchor的权重赋1

        ret_dict = {
            'box_cls_labels': labels, # (107136,)
            'box_reg_targets': bbox_targets,  # (107136,7) 编码后的结果
            'reg_weights': reg_weights, # (107136,)
        }
        return ret_dict
