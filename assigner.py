import torch
from iou import IoU


class Assigner:
    def __init__(self, config):
        super(Assigner, self).__init__()
        self.pos_iou_thr = config['pos_iou_thr']
        self.neg_iou_thr = config['neg_iou_thr']
        self.min_pos_iou = config['min_pos_iou']

        self.ignore_iof_thr = config['ignore_iof_thr']
        self.match_low_quality = config['match_low_quality']
        self.iou_calculator = IoU()

    def assign(self, bboxes, gt_bboxes, gt_labels=None):
        num_bboxes = bboxes.size()[0]
        num_gts = gt_bboxes.size()[0]

        overlaps = self.iou_calculator(gt_bboxes, bboxes)

        assigned_gt_inds = overlaps.new_full((num_bboxes, ), -1, dtype=torch.long)

        max_overlaps, argmax_overlaps = overlaps.max(dim=0)

        if not self.match_low_quality:
            if isinstance(self.neg_iou_thr, float):
                assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps < self.neg_iou_thr)] = 0

            pos_inds = max_overlaps >= self.pos_iou_thr
            assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        else:
            gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)
            for i in range(num_gts):
                if gt_max_overlaps[i] >= self.min_pos_iou:
                    max_iou_inds = overlaps[i, :]==gt_max_overlaps[i]
                    assigned_gt_inds[max_iou_inds] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)


class AssignResult:
    def __init__(self, num_gts, gt_inds, max_overlaps, labels):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels

    def add_gt_(self, gt_labels):
        self_inds = torch.arange(1, len(gt_labels) + 1, dtype=torch.long, device=gt_labels.device)
        self.gt_inds = torch.cat([self_inds, self.gt_inds])
        self.max_overlaps = torch.cat([self.max_overlaps.new_ones(len(gt_labels)), self.max_overlaps])
        if self.labels is not None:
            self.labels = torch.cat([gt_labels, self.labels])
