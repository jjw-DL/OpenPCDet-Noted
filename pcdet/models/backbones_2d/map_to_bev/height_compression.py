import torch.nn as nn


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        """
        在高度方向上进行压缩
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES # 256

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        # 结合batch，spatial_shape、indice和feature将特征还原的对应位置
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape # 4，128，2，200，150
        spatial_features = spatial_features.view(N, C * D, H, W) # （4，256，200，150）在高度方向上合并，将特征图压缩至BEV特征图
        # 将特征和采样尺度加入batch_dict
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride'] # 8
        return batch_dict
