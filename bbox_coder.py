import torch
import numpy as np


class BBoxCoder:
    def __init__(self, config):
        super(BBoxCoder, self).__init__()
        self.means = config['target_means']
        self.stds = config['target_stds']

    def encode(self, bboxes, gt_bboxes):
        encoded_bboxes = self.bbox2delta(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

    def decode(self, bboxes, pred_bboxes, max_shape=None, wh_ratio_clip=16 / 1000):
        decoded_bboxes = self.delta2bbox(bboxes, pred_bboxes, self.means, self.stds, max_shape, wh_ratio_clip)
        return decoded_bboxes

    @staticmethod
    def bbox2delta(proposals, gt, means=(0., 0., 0., 0.), stds=(1., 1., 1., 1.)):
        proposals = proposals.float()
        gt = gt.float()
        px = (proposals[..., 0] + proposals[..., 2]) * 0.5
        py = (proposals[..., 1] + proposals[..., 3]) * 0.5
        pw = proposals[..., 2] - proposals[..., 0]
        ph = proposals[..., 3] - proposals[..., 1]

        gx = (gt[..., 0] + gt[..., 2]) * 0.5
        gy = (gt[..., 1] + gt[..., 3]) * 0.5
        gw = gt[..., 2] - gt[..., 0]
        gh = gt[..., 3] - gt[..., 1]

        dx = (gx - px) / pw
        dy = (gy - py) / ph
        dw = torch.log(gw / pw)
        dh = torch.log(gh / ph)
        deltas = torch.stack([dx, dy, dw, dh], dim=-1)

        means = deltas.new_tensor(means).unsqueeze(0)
        stds = deltas.new_tensor(stds).unsqueeze(0)
        deltas = deltas.sub_(means).div_(stds)

        return deltas

    @staticmethod
    def delta2bbox(rois, deltas, means=(0., 0., 0., 0.), stds=(1., 1., 1., 1.), max_shape=None, wh_ratio_clip=16 / 1000):
        num_bboxes, num_classes = deltas.size(0), deltas.size(1) // 4
        if num_bboxes == 0:
            return deltas

        deltas = deltas.reshape(-1, 4)

        means = deltas.new_tensor(means).view(1, -1)
        stds = deltas.new_tensor(stds).view(1, -1)
        denorm_deltas = deltas * stds + means

        dxy = denorm_deltas[:, :2]
        dwh = denorm_deltas[:, 2:]

        rois_ = rois.repeat(1, num_classes).reshape(-1, 4)
        pxy = ((rois_[:, :2] + rois_[:, 2:]) * 0.5)
        pwh = (rois_[:, 2:] - rois_[:, :2])

        dxy_wh = pwh * dxy

        max_ratio = np.abs(np.log(wh_ratio_clip))

        dwh = dwh.clamp(min=-max_ratio, max=max_ratio)

        gxy = pxy + dxy_wh
        gwh = pwh * dwh.exp()
        x1y1 = gxy - (gwh * 0.5)
        x2y2 = gxy + (gwh * 0.5)
        bboxes = torch.cat([x1y1, x2y2], dim=-1)
        if max_shape is not None:
            bboxes[..., 0::2].clamp_(min=0, max=max_shape[1])
            bboxes[..., 1::2].clamp_(min=0, max=max_shape[0])
        bboxes = bboxes.reshape(num_bboxes, -1)
        return bboxes
