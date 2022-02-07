import numpy as np
import torch
import torch.nn as nn

from ....ops.iou3d_nms import iou3d_nms_utils


class ProposalTargetLayer(nn.Module):
    def __init__(self, roi_sampler_cfg):
        super().__init__()
        self.roi_sampler_cfg = roi_sampler_cfg # TARGET_CONFIG:10个配置参数

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:
            batch_dict:
                rois: (B, M, 7 + C)
                gt_of_rois: (B, M, 7 + C)
                gt_iou_of_rois: (B, M)
                roi_scores: (B, M)
                roi_labels: (B, M)
                reg_valid_mask: (B, M)
                rcnn_cls_labels: (B, M)
        """
        # batch_rois 采样后的ROI:(8,128,7)
        # batch_gt_of_rois 采样后ROI对应的gt(8,128,8)
        # batch_roi_ious 采样后ROI与gt的最大iou (8,128)
        # batch_roi_scores 采样后ROI分数 (8,128）
        # batch_roi_labels 采样后ROI标签 (8,128)
        batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels = self.sample_rois_for_rcnn(
            batch_dict=batch_dict
        )
        # 根据iou继续过滤
        # regression valid mask
        reg_valid_mask = (batch_roi_ious > self.roi_sampler_cfg.REG_FG_THRESH).long()

        # classification label
        if self.roi_sampler_cfg.CLS_SCORE_TYPE == 'cls':
            batch_cls_labels = (batch_roi_ious > self.roi_sampler_cfg.CLS_FG_THRESH).long()
            ignore_mask = (batch_roi_ious > self.roi_sampler_cfg.CLS_BG_THRESH) & \
                          (batch_roi_ious < self.roi_sampler_cfg.CLS_FG_THRESH)
            batch_cls_labels[ignore_mask > 0] = -1
        elif self.roi_sampler_cfg.CLS_SCORE_TYPE == 'roi_iou':
            iou_bg_thresh = self.roi_sampler_cfg.CLS_BG_THRESH # 0.75
            iou_fg_thresh = self.roi_sampler_cfg.CLS_FG_THRESH # 0.25
            fg_mask = batch_roi_ious > iou_fg_thresh # 前景
            bg_mask = batch_roi_ious < iou_bg_thresh # 背景
            interval_mask = (fg_mask == 0) & (bg_mask == 0) # ignore

            batch_cls_labels = (fg_mask > 0).float() # 前景标签（0和1）
            batch_cls_labels[interval_mask] = \
                (batch_roi_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh) # 计算interval box的标签（一个iou的相对值）
        else:
            raise NotImplementedError

        targets_dict = {'rois': batch_rois, 'gt_of_rois': batch_gt_of_rois, 'gt_iou_of_rois': batch_roi_ious,
                        'roi_scores': batch_roi_scores, 'roi_labels': batch_roi_labels,
                        'reg_valid_mask': reg_valid_mask, # 回归mask (8, 128)
                        'rcnn_cls_labels': batch_cls_labels} # 类别标签 (8, 128)

        return targets_dict

    def sample_rois_for_rcnn(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:

        """
        # 1.提取所需信息
        batch_size = batch_dict['batch_size'] # 8
        rois = batch_dict['rois'] # (8,512,7)
        roi_scores = batch_dict['roi_scores'] # (8,512)
        roi_labels = batch_dict['roi_labels'] # (8,512)
        gt_boxes = batch_dict['gt_boxes'] # (8,17,8)
        # 2.初始化
        code_size = rois.shape[-1] #  7
        batch_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size) # (8,128,7)
        batch_gt_of_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size + 1) # (8,128,8)
        batch_roi_ious = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE) # (8,128)
        batch_roi_scores = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE) # (8, 128)
        batch_roi_labels = rois.new_zeros((batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE), dtype=torch.long) # (8, 128)
        # 3.逐帧处理
        for index in range(batch_size):
            # 根据batch index提取当前帧的rio相关和gt信息
            cur_roi, cur_gt, cur_roi_labels, cur_roi_scores = \
                rois[index], gt_boxes[index], roi_labels[index], roi_scores[index]
            # 过滤掉空gt
            k = cur_gt.__len__() - 1
            while k > 0 and cur_gt[k].sum() == 0:
                k -= 1
            cur_gt = cur_gt[:k + 1] # 加1的原因是区间前闭后开
            cur_gt = cur_gt.new_zeros((1, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt # 避免gt为0

            if self.roi_sampler_cfg.get('SAMPLE_ROI_BY_EACH_CLASS', False):
                # 3.1求与ROI最匹配的gt的iou和索引(逐类别处理)
                max_overlaps, gt_assignment = self.get_max_iou_with_same_class(
                    rois=cur_roi, roi_labels=cur_roi_labels,
                    gt_boxes=cur_gt[:, 0:7], gt_labels=cur_gt[:, -1].long()
                ) # (512,) (512,)
            else:
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7])  # (M, N)
                max_overlaps, gt_assignment = torch.max(iou3d, dim=1)

            # 3.2 计算前景和背景的采样索引 128
            sampled_inds = self.subsample_rois(max_overlaps=max_overlaps)

            # 3.3 根据采样索引提取roi相关信息（逐帧赋值）
            batch_rois[index] = cur_roi[sampled_inds] # roi的box
            batch_roi_labels[index] = cur_roi_labels[sampled_inds] # roi的标签
            batch_roi_ious[index] = max_overlaps[sampled_inds] # roi与gt的iou
            batch_roi_scores[index] = cur_roi_scores[sampled_inds] # roi的预测分数
            batch_gt_of_rois[index] = cur_gt[gt_assignment[sampled_inds]] # roi对应的gt

        return batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels

    def subsample_rois(self, max_overlaps):
        """
        sample fg, easy_bg, hard_bg
        Args:
            max_overlaps: roi与gt最大iou
        Returns:

        """
        # 1.计算采样个数和阈值并根据阈值分配前景和背景
        fg_rois_per_image = int(np.round(self.roi_sampler_cfg.FG_RATIO * self.roi_sampler_cfg.ROI_PER_IMAGE)) # 64 = 128 * 0.5
        fg_thresh = min(self.roi_sampler_cfg.REG_FG_THRESH, self.roi_sampler_cfg.CLS_FG_THRESH) # 0.55 <-- min(0.55, 0.75) 取回归和分类的最小阈值

        fg_inds = ((max_overlaps >= fg_thresh)).nonzero().view(-1) # 计算前景索引(iou > thresh)
        easy_bg_inds = ((max_overlaps < self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1) # 计算简单背景索引 (iou < thresh low)
        hard_bg_inds = ((max_overlaps < self.roi_sampler_cfg.REG_FG_THRESH) &
                (max_overlaps >= self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1) # 计算困难背景的索引

        fg_num_rois = fg_inds.numel() # 计算前景的个数
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel() # 计算背景的个数
        # 2.分四种情况采样
        # 2.1 前景和背景都大于0
        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE - fg_rois_per_this_image
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )
        # 2.2 前景大于0，背景等于0
        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(self.roi_sampler_cfg.ROI_PER_IMAGE) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num]
            bg_inds = []
        # 2.3 前景等于0，背景大于0
        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )
        # 2.4 前景和背景都等于0
        else:
            # 打印iou的最大值和最小值，以及前景和背景的数量(这种情况应该不存在)
            print('maxoverlaps:(min=%f, max=%f)' % (max_overlaps.min().item(), max_overlaps.max().item()))
            print('ERROR: FG=%d, BG=%d' % (fg_num_rois, bg_num_rois))
            raise NotImplementedError

        sampled_inds = torch.cat((fg_inds, bg_inds), dim=0) # 将前景和背景的采样索引拼接返回（采样128个）
        return sampled_inds

    @staticmethod
    def sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, hard_bg_ratio):
        # 和上面的采样类似，前面是针对前景和背景采样，这里是针对背景的easy和hard采样
        if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
            hard_bg_rois_num = min(int(bg_rois_per_this_image * hard_bg_ratio), len(hard_bg_inds)) # 计算hard bg的采样数量
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num # 计算easy bg的采样数量

            # sampling hard bg
            # 在0到hard bg总量之间生成hard_bg_rois_num个随机数eg：0-200生成100个随机数
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            hard_bg_inds = hard_bg_inds[rand_idx]

            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            easy_bg_inds = easy_bg_inds[rand_idx]

            bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0) # 将采样的bg进行拼接
        elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            bg_inds = hard_bg_inds[rand_idx]
        elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            bg_inds = easy_bg_inds[rand_idx]
        else:
            raise NotImplementedError

        return bg_inds

    @staticmethod
    def get_max_iou_with_same_class(rois, roi_labels, gt_boxes, gt_labels):
        """
        Args:
            rois: (N, 7)
            roi_labels: (N)
            gt_boxes: (M, 7)
            gt_labels: (M, )

        Returns:
            与ROI最匹配的gt的iou和索引(逐类别处理)
        """
        max_overlaps = rois.new_zeros(rois.shape[0]) # (512,)
        gt_assignment = roi_labels.new_zeros(roi_labels.shape[0]) # (512,)
        # 逐类别处理
        for k in range(gt_labels.min().item(), gt_labels.max().item() + 1):
            # 计算roi和gt关于该类别的mask
            roi_mask = (roi_labels == k)
            gt_mask = (gt_labels == k)
            if roi_mask.sum() > 0 and gt_mask.sum() > 0:
                cur_roi = rois[roi_mask]
                cur_gt = gt_boxes[gt_mask]
                original_gt_assignment = gt_mask.nonzero().view(-1) # 该类别gt在原始gt中的位置
                
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt)  # 调用函数计算roi和gt的iou --> (512, 17)
                cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1) # 找到每个roi最匹配gt的iou和索引
                max_overlaps[roi_mask] = cur_max_overlaps # 将与最匹配gt的iou赋值到对应位置
                gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment] # 将与最匹配gt的索引赋值到对应位置(二次索引)
        
        return max_overlaps, gt_assignment
