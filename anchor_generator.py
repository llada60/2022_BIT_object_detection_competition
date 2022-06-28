import torch
import numpy as np


class AnchorGenerator:
    def __init__(self, config):
        self.strides = [(stride, stride) for stride in config['strides']]
        self.base_sizes = [min(stride) for stride in self.strides]
        self.scales = torch.Tensor(config['scales'])
        self.ratios = torch.Tensor(config['ratios'])
        self.scale_major = True
        self.center_offset = 0.0
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_priors(self):
        return [base_anchors.size(0) for base_anchors in self.base_anchors]

    @property
    def num_levels(self):
        return len(self.strides)

    def gen_base_anchors(self):
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(
                    base_size=base_size,
                    scales=self.scales,
                    ratios=self.ratios))
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self, base_size, scales, ratios):
        w = base_size
        h = base_size
        x_center = self.center_offset * w
        y_center = self.center_offset * h

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios
        ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
            y_center + 0.5 * hs
        ]
        base_anchors = torch.stack(base_anchors, dim=-1)

        return base_anchors

    @staticmethod
    def _meshgrid(x, y, row_major=True):
        xx = x.repeat(y.shape[0])
        yy = y.view(-1, 1).repeat(1, x.shape[0]).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_priors(self, featmap_sizes, dtype=torch.float32, device='cpu'):
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_priors(featmap_sizes[i], level_idx=i, dtype=dtype, device=device)
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_priors(self, featmap_size, level_idx, dtype=torch.float32, device='cuda'):
        base_anchors = self.base_anchors[level_idx].to(device).to(dtype)
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]

        shift_x = torch.arange(0, feat_w, device=device).to(dtype) * stride_w
        shift_y = torch.arange(0, feat_h, device=device).to(dtype) * stride_h

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)

        return all_anchors

    def valid_flags(self, featmap_sizes, pad_shape, device='cuda'):
        multi_level_flags = []
        for i in range(self.num_levels):
            anchor_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / anchor_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / anchor_stride[0])), feat_w)
            flags = self.single_level_valid_flags((feat_h, feat_w),
                                                  (valid_feat_h, valid_feat_w),
                                                  self.num_base_priors[i],
                                                  device=device)
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(self, featmap_size, valid_size, num_base_anchors, device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size

        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(valid.size(0), num_base_anchors).contiguous().view(-1)
        return valid
