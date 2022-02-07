from .detector3d_template import Detector3DTemplate


class VoxelRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks() # 各模块的instance

    def forward(self, batch_dict):
        # 逐模块调用forward函数
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        # 如果在训练模式下，则获取loss
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        # 在测试模式下，对预测结果进行后处理
        else:
            """
            pred_dicts = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            recall_dicts:根据全部训练数据得到的召回率
            """
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        # 计算rpn loss，tb_dict是一个字典包含了{'rpn_loss_cls', 'rpn_loss_loc', 'rpn_loss_dir'}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        # rcnn loss
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict
