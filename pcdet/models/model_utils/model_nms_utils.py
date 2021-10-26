import torch

from ...ops.iou3d_nms import iou3d_nms_utils


def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):
    # 1.首先根据置信度阈值过滤掉部分box
    src_box_scores = box_scores
    if score_thresh is not None:
        scores_mask = (box_scores >= score_thresh)
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]

    selected = []
    if box_scores.shape[0] > 0:
        # 如果剩余的box数量大于0.则选出分数前k个box
        box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
        boxes_for_nms = box_preds[indices]
        # 调用iou3d_nms_utils的nms_gpu函数进行nms，返回的是被保留下的box的索引，selected_scores = None
        # 这里传入的box已经是根据分数进行排序的，但是在nms_gpu又进行了一次排序(感觉有点重复)
        keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
        )
        # 根据返回索引找出box索引值
        selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

    if score_thresh is not None:
        # 如果存在置信读阈值，scores_mask是box_scores在src_box_scores中的索引，即原始索引
        original_idxs = scores_mask.nonzero().view(-1)
        # selected表示的box_scores的选择索引，经过这次索引，selected表示的是src_box_scores被选择的索引
        selected = original_idxs[selected]
        # 总结：双重索引，值得学习
    return selected, src_box_scores[selected]


def multi_classes_nms(cls_scores, box_preds, nms_config, score_thresh=None):
    """
    Args:
        cls_scores: (N, num_class)
        box_preds: (N, 7 + C)
        nms_config:
        score_thresh:

    Returns:

    """
    pred_scores, pred_labels, pred_boxes = [], [], []
    # 逐类别进行NMS
    for k in range(cls_scores.shape[1]):
        if score_thresh is not None:
            # 如果分数阈值不空，则首先过滤掉分数过低的box
            scores_mask = (cls_scores[:, k] >= score_thresh)
            box_scores = cls_scores[scores_mask, k]
            cur_box_preds = box_preds[scores_mask]
        else:
            box_scores = cls_scores[:, k]
            cur_box_preds = box_preds

        selected = []
        if box_scores.shape[0] > 0:
            # 选出前k个分数值和索引
            box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
            # 获取前k个预测box
            boxes_for_nms = cur_box_preds[indices]
            # nms_config.NMS_TYPE: nms_gpu
            # 输入为box，box分数和NMS阈值
            # 输出为保留下的索引
            keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                    boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
            )
            # 取出前NMS_POST_MAXSIZE=500作为被选择的
            selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]
        # 筛选被选择的box
        pred_scores.append(box_scores[selected])
        pred_labels.append(box_scores.new_ones(len(selected)).long() * k) # 预测类别都是k
        pred_boxes.append(cur_box_preds[selected])
    # 将预测结果拼接
    pred_scores = torch.cat(pred_scores, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    pred_boxes = torch.cat(pred_boxes, dim=0)

    return pred_scores, pred_labels, pred_boxes
