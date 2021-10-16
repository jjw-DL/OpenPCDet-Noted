import numpy as np


class PointFeatureEncoder(object):
    """
    该类决定使用点的哪些属性 比如x,y,z等
    """
    def __init__(self, config, point_cloud_range=None):
        super().__init__()
        self.point_encoding_config = config
        assert list(self.point_encoding_config.src_feature_list[0:3]) == ['x', 'y', 'z']
        self.used_feature_list = self.point_encoding_config.used_feature_list # ['x', 'y', 'z', 'intensity']
        self.src_feature_list = self.point_encoding_config.src_feature_list # ['x', 'y', 'z', 'intensity']
        self.point_cloud_range = point_cloud_range

    @property
    def num_point_features(self):
        # encoding_type: absolute_coordinates_encoding
        return getattr(self, self.point_encoding_config.encoding_type)(points=None) # 4

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                ...
        Returns:
            data_dict:
                points: (N, 3 + C_out),
                use_lead_xyz: whether to use xyz as point-wise features
                ...
        """
        # (N, 4) , True
        data_dict['points'], use_lead_xyz = getattr(self, self.point_encoding_config.encoding_type)(
            data_dict['points']
        )
        data_dict['use_lead_xyz'] = use_lead_xyz # True
        return data_dict

    def absolute_coordinates_encoding(self, points=None):
        if points is None:
            num_output_features = len(self.used_feature_list)
            return num_output_features

        point_feature_list = [points[:, 0:3]] # (1, N, 3 + C_in)
        for x in self.used_feature_list:
            if x in ['x', 'y', 'z']:
                continue
            idx = self.src_feature_list.index(x) # 3
            point_feature_list.append(points[:, idx:idx+1]) # [(1, N, 3), (N, 1)]
        point_features = np.concatenate(point_feature_list, axis=1) # (N, 4)
        return point_features, True
