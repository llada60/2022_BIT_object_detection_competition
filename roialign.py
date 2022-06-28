import torch.nn as nn
from mmcv.ops import roi_align


class RoIAlign(nn.Module):
    def __init__(self, config, spatial_scale):
        super(RoIAlign, self).__init__()
        self.output_size = (config['output_size'], config['output_size'])
        self.spatial_scale = spatial_scale
        self.sampling_ratio = int(config['sampling_ratio'])
        self.pool_mode = 'avg'
        self.aligned = True

    def forward(self, input, rois):
        return roi_align(input, rois, self.output_size, self.spatial_scale, self.sampling_ratio, self.pool_mode,
                         self.aligned)
