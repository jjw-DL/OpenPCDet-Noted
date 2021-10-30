import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


def statistics_info(cfg, ret_dict, metric, disp_dict):
    """
    统计信息
    Args:
        cfg:配置文件
        ret_dict:结果字典
        metric:度量字典
        disp_dict:展示字典
    """
    # [0.3,0.5,0.7]根据不同的阈值进行累加
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0) # 真值框的数量
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0] # 0.3
    # 最小阈值的展示字典统计
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    """
    模型评估
    Args:
        cfg:配置文件
        model:模型
        dataloader: 数据加载器
        epoch_id：epoch的id
        logger:日志记录器
        dist_test:分布式测试
        save_to_file: 保存到文件
        result_dir: 结果文件夹:OpenPCDet/output/kitti_models/pointpillar/default/eval/epoch_80/val/default
    Returns:
        ret_dict: 结果字典
    """
    # 构造文件夹
    result_dir.mkdir(parents=True, exist_ok=True)
    # /home/ggj/ObjectDetection/OpenPCDet/output/kitti_models/pointpillar/default/eval/epoch_80/val/default/final_result/data
    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    # 初始化后处理metric阈值字典
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST: # [0.3, 0.5, 0.7]
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names # ['Car', 'Pedestrian', 'Cyclist']
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    # 如果是分布式测试，则需要处理以下model
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    # 将model设置为eval模式
    model.eval()
    # 单GPU模式下创建进度条
    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    # 记录开始时间
    start_time = time.time()
    # 开始评估
    for i, batch_dict in enumerate(dataloader):
        # 1.将数据加载到GPU
        load_data_to_gpu(batch_dict)
        # 2.在no_grad模式下将数据通过model前传
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        # 初始化展示字典
        disp_dict = {}
        # 3.统计累计metric信息
        statistics_info(cfg, ret_dict, metric, disp_dict)
        # 4.将预测结果转换到自定义的坐标系下
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        # 5.将检测的结果加入检测列表
        det_annos += annos
        # 更新进度条
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()
    # 评估结束后关闭进度条
    if cfg.LOCAL_RANK == 0:
        progress_bar.close()
    # 在分布式测试模式下，需要将结果统一到master节点上
    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    # 6.计算每帧点云的平均预测时间
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    # 总共预测的物体数量
    gt_num_cnt = metric['gt_num']
    # 7.累加结果/总数量，计算百分比
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    # 8.计算平均每帧预测多少物体
    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    # 将结果写入文件夹
    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    # 9.将预测结果放入dataset.evaluation计算指标并打印
    # dataset.evaluation()-->kitti_eval.get_official_eval_result()-->eval()
    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    # 记录结果：AP或AP_R40，overlap=0.7或0.5
    logger.info(result_str)
    # 更新结果字典
    ret_dict.update(result_dict) 

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
