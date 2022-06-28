import torch
import torch.nn as nn
from utils import multi_apply
import torch.nn.functional as F
from bbox_coder import BBoxCoder
from mmdet.core.post_processing import multiclass_nms
from mmdet.models.losses import SmoothL1Loss, CrossEntropyLoss


class BBoxHead(nn.Module):
    def __init__(self, config):
        super(BBoxHead, self).__init__()
        self.roi_feat_size = (config['roi_feat_size'], config['roi_feat_size'])
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.num_classes = 21
        self.in_channels = config['in_channels'] * self.roi_feat_area
        self.num_shared_fcs = 2
        self.fc_out_channels = config['fc_out_channels']

        self.bbox_coder = BBoxCoder(config['bbox_coder'])
        self.loss_cls = CrossEntropyLoss(use_sigmoid=config['loss_cls']['use_sigmoid'], loss_weight=config['loss_cls']['loss_weight'])
        self.loss_bbox = SmoothL1Loss(beta=config['loss_bbox']['beta'], loss_weight=config['loss_bbox']['loss_weight'])

        self.shared_fcs = nn.Sequential()

        for i in range(self.num_shared_fcs):
            fc_in_channels = (self.in_channels if i == 0 else self.fc_out_channels)
            self.shared_fcs.append(
                nn.Linear(
                    in_features=fc_in_channels,
                    out_features=self.fc_out_channels)
            )
            self.shared_fcs.append(
                nn.ReLU(inplace=True)
            )

        self.fc_cls = nn.Linear(
            in_features=self.fc_out_channels,
            out_features=self.num_classes + 1
        )

        self.fc_reg = nn.Linear(
            in_features=self.fc_out_channels,
            out_features=4
        )

    def forward(self, x):
        x = x.flatten(1)
        x = self.shared_fcs(x)

        x_cls = x
        x_reg = x

        if x_cls.dim() > 2:
            x_cls = x_cls.flatten(1)
        if x_reg.dim() > 2:
            x_reg = x_reg.flatten(1)

        cls_score = self.fc_cls(x_cls)
        bbox_pred = self.fc_reg(x_reg)

        return cls_score, bbox_pred

    def loss(self, cls_score, bbox_pred, rois, labels, label_weights, bbox_targets, bbox_weights, reduction_override=None):
        losses = dict()

        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                losses['acc'] = self.accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            if pos_inds.any():
                pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes, pos_gt_labels, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        labels = pos_bboxes.new_full((num_samples, ), self.num_classes, dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg['pos_weight'] <= 0 else cfg['pos_weight']
            label_weights[:num_pos] = pos_weight
            pos_bbox_targets = self.bbox_coder.encode(pos_bboxes, pos_gt_bboxes)
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self, sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg, concat=True):
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights

    @staticmethod
    def accuracy(pred, target, topk=1):
        if isinstance(topk, int):
            topk = (topk,)
            return_single = True
        else:
            return_single = False

        maxk = max(topk)
        if pred.size(0) == 0:
            accu = [pred.new_tensor(0.) for i in range(len(topk))]
            return accu[0] if return_single else accu

        pred_value, pred_label = pred.topk(maxk, dim=1)
        pred_label = pred_label.t()
        correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / pred.size(0)))
        return res[0] if return_single else res

    def get_bboxes(self, rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=False, cfg=None):
        scores = F.softmax(
            cls_score, dim=-1) if cls_score is not None else None
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg['score_thr'], cfg['nms'],
                                                    cfg['max_per_img'])

            return det_bboxes, det_labels

    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() <= len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(
                rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_, img_meta_)

            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds.type(torch.bool)])

        return bboxes_list

    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        max_shape = img_meta['img_shape']

        if rois.size(1) == 4:
            new_rois = self.bbox_coder.decode(
                rois, bbox_pred, max_shape=max_shape)
        else:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=max_shape)
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois
