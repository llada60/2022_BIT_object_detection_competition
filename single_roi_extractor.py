import torch
import torch.nn as nn
from roialign import RoIAlign


class SingleRoIExtractor(nn.Module):
    def __init__(self, config):
        super(SingleRoIExtractor, self).__init__()
        self.roi_layers = self.build_roi_layers(config['roi_layer'], config['featmap_strides'])
        self.out_channels = config['out_channels']
        self.featmap_strides = config['featmap_strides']
        self.finest_scale = 56

    @staticmethod
    def build_roi_layers(layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        roi_layers = nn.ModuleList([RoIAlign(cfg, 1 / s) for s in featmap_strides])
        return roi_layers

    def map_roi_levels(self, rois, num_levels):
        scale = torch.sqrt((rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def forward(self, feats, rois):
        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)

        roi_feats = feats[0].new_zeros(rois.size(0), self.out_channels, *out_size)

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)

        target_lvls = self.map_roi_levels(rois, num_levels)

        for i in range(num_levels):
            mask = target_lvls == i
            inds = mask.nonzero(as_tuple=False).squeeze(1)
            if inds.numel() > 0:
                rois_ = rois[inds]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t
            else:
                roi_feats += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.
        return roi_feats

    @property
    def num_inputs(self):
        return len(self.featmap_strides)
