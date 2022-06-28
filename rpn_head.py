from utils import *
import torch.nn as nn
from nms import batched_nms
from assigner import Assigner
import torch.nn.functional as F
from bbox_coder import BBoxCoder
from random_sampler import RandomSampler
from anchor_generator import AnchorGenerator
from mmdet.models.losses import SmoothL1Loss, CrossEntropyLoss


class RPNHead(nn.Module):
    def __init__(self, config, train_cfg, test_cfg):
        super(RPNHead, self).__init__()
        self.num_classes = 21
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.in_channels = config['in_channels']
        self.feat_channels = config['feat_channels']

        self.rpn_conv = nn.Conv2d(
            in_channels=config['in_channels'],
            out_channels=config['feat_channels'],
            kernel_size=3,
            padding=1
        )
        self.rpn_cls = nn.Conv2d(
            in_channels=config['feat_channels'],
            out_channels=3,
            kernel_size=1
        )
        self.rpn_reg = nn.Conv2d(
            in_channels=config['feat_channels'],
            out_channels=12,
            kernel_size=1
        )

        self.bbox_coder = BBoxCoder(config['bbox_coder'])
        self.loss_cls = CrossEntropyLoss(use_sigmoid=config['loss_cls']['use_sigmoid'],
                                         loss_weight=config['loss_cls']['loss_weight'])
        self.loss_bbox = SmoothL1Loss(beta=config['loss_bbox']['beta'], loss_weight=config['loss_bbox']['loss_weight'])
        self.prior_generator = AnchorGenerator(config['anchor_generator'])

        self.assigner = Assigner(self.train_cfg['rpn']['assigner'])
        self.sampler = RandomSampler(self.train_cfg['rpn']['sampler'])

        self.use_sigmoid_cls = config['loss_cls']['use_sigmoid']

    def forward_single(self, x):
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_train(self, x, img_metas, gt_bboxes, proposal_cfg=None):
        outs = self(x)
        loss_inputs = outs + (gt_bboxes, img_metas)
        losses = self.loss(*loss_inputs)
        proposal_list = self.get_bboxes(*outs, img_metas=img_metas, cfg=proposal_cfg)
        return losses, proposal_list

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, 1)
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples)

        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)

        loss_bbox = self.loss_bbox(bbox_pred, bbox_targets, bbox_weights, avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    def loss(self, cls_scores, bbox_preds, gt_bboxes, img_metas):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        cls_reg_targets = self.get_targets(anchor_list, valid_flag_list, gt_bboxes, img_metas)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = num_total_pos + num_total_neg

        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        num_imgs = len(img_metas)

        multi_level_anchors = self.prior_generator.grid_priors(featmap_sizes, device=device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.prior_generator.valid_flags(featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def _get_targets_single(self, flat_anchors, valid_flags, gt_bboxes, img_meta):
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags, img_meta['img_shape'][:2])
        if not inside_flags.any():
            return (None,) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        assign_result = self.assigner.assign(anchors, gt_bboxes)
        sampling_result = self.sampler.sample(assign_result, anchors, gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,), self.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            labels[pos_inds] = 0
            if self.train_cfg['rpn']['pos_weight'] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg['rpn']['pos_weight']
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        num_total_anchors = flat_anchors.size(0)
        labels = unmap(
            labels, num_total_anchors, inside_flags,
            fill=self.num_classes)  # fill bg label
        label_weights = unmap(label_weights, num_total_anchors,
                              inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)

    def get_targets(self, anchor_list, valid_flag_list, gt_bboxes_list, img_metas):
        num_imgs = len(img_metas)

        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            img_metas
        )
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)

        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg)

        return res

    def get_bboxes(self, cls_scores,  bbox_preds, img_metas=None, cfg=None):
        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device
        )

        result_list = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)

            results = self._get_bboxes_single(cls_score_list, bbox_pred_list, mlvl_priors, img_meta, cfg)
            result_list.append(results)
        return result_list

    def _get_bboxes_single(self, cls_score_list, bbox_pred_list, mlvl_anchors, img_meta, cfg):
        cfg = self.train_cfg['rpn_proposal'] if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']

        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        nms_pre = cfg.get('nms_pre', -1)
        for level_idx in range(len(cls_score_list)):
            rpn_cls_score = cls_score_list[level_idx]
            rpn_bbox_pred = bbox_pred_list[level_idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 0]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            anchors = mlvl_anchors[level_idx]
            if 0 < nms_pre < scores.shape[0]:
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:nms_pre]
                scores = ranked_scores[:nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]

            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(scores.new_full((scores.size(0),), level_idx, dtype=torch.long))

        return self._bbox_post_process(mlvl_scores, mlvl_bbox_preds,
                                       mlvl_valid_anchors, level_ids, cfg,
                                       img_shape)

    def _bbox_post_process(self, mlvl_scores, mlvl_bboxes, mlvl_valid_anchors, level_ids, cfg, img_shape):
        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bboxes)
        proposals = self.bbox_coder.decode(anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        w = proposals[:, 2] - proposals[:, 0]
        h = proposals[:, 3] - proposals[:, 1]

        valid_mask = (w > cfg['min_bbox_size']) & (h > cfg['min_bbox_size'])
        if not valid_mask.all():
            proposals = proposals[valid_mask]
            scores = scores[valid_mask]
            ids = ids[valid_mask]

        if proposals.numel() > 0:
            dets, _ = batched_nms(proposals, scores, ids, cfg['nms'])
        else:
            return proposals.new_zeros(0, 5)

        return dets[:cfg['max_per_img']]

    def simple_test_rpn(self, x, img_metas):
        rpn_outs = self(x)
        proposal_list = self.get_bboxes(*rpn_outs, img_metas=img_metas)
        return proposal_list

    def aug_test_rpn(self, feats, img_metas):
        samples_per_gpu = len(img_metas[0])
        aug_proposals = [[] for _ in range(samples_per_gpu)]
        for x, img_meta in zip(feats, img_metas):
            proposal_list = self.simple_test_rpn(x, img_meta)
            for i, proposals in enumerate(proposal_list):
                aug_proposals[i].append(proposals)

        aug_img_metas = []
        for i in range(samples_per_gpu):
            aug_img_meta = []
            for j in range(len(img_metas)):
                aug_img_meta.append(img_metas[j][i])
            aug_img_metas.append(aug_img_meta)

        merged_proposals = [
            merge_aug_proposals(proposals, aug_img_meta, self.test_cfg['rpn'])
            for proposals, aug_img_meta in zip(aug_proposals, aug_img_metas)
        ]
        return merged_proposals


def merge_aug_proposals(aug_proposals, img_metas, cfg):
    cfg = copy.deepcopy(cfg)
    recovered_proposals = []
    for proposals, img_info in zip(aug_proposals, img_metas):
        img_shape = img_info['img_shape']
        scale_factor = img_info['scale_factor']
        flip = img_info['flip']
        flip_direction = img_info['flip_direction']
        _proposals = proposals.clone()
        _proposals[:, :4] = bbox_mapping_back(_proposals[:, :4], img_shape,
                                              scale_factor, flip,
                                              flip_direction)
        recovered_proposals.append(_proposals)
    aug_proposals = torch.cat(recovered_proposals, dim=0)
    merged_proposals, _ = nms(aug_proposals[:, :4].contiguous(),
                              aug_proposals[:, -1].contiguous(),
                              cfg['nms']['iou_threshold'])
    scores = merged_proposals[:, 4]
    _, order = scores.sort(0, descending=True)
    num = min(cfg['max_per_img'], merged_proposals.shape[0])
    order = order[:num]
    merged_proposals = merged_proposals[order, :]
    return merged_proposals
