import torch
from fpn import FPN
import torch.nn as nn
from resnet import ResNet50
from rpn_head import RPNHead
from collections import OrderedDict
from cascade_roi_head import CascadeRoIHead


class CascadeRCNN(nn.Module):
    def __init__(self, config):
        super(CascadeRCNN, self).__init__()
        self.backbone = ResNet50(config['backbone'])
        self.neck = FPN(config['neck'])
        self.rpn_head = RPNHead(config['rpn_head'], train_cfg=config['train_cfg'], test_cfg=config['test_cfg'])
        self.roi_head = CascadeRoIHead(config['roi_head'], config['train_cfg']['rcnn'], config['test_cfg']['rcnn'])

        self.train_cfg = config['train_cfg']
        self.test_cfg = config['test_cfg']

    def extract_feat(self, img):
        x = self.backbone(img)
        x = self.neck(x)
        return x

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        return [self.extract_feat(img) for img in imgs]

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels):
        x = self.extract_feat(img)
        rpn_losses, proposal_list = self.rpn_head.forward_train(x, img_metas, gt_bboxes,
                                                                proposal_cfg=self.train_cfg['rpn_proposal'])
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list, gt_bboxes, gt_labels)

        losses = dict()
        losses.update(rpn_losses)
        losses.update(roi_losses)
        return losses

    def forward_test(self, imgs, img_metas, **kwargs):
        num_augs = len(imgs)
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(x, proposal_list, img_metas, rescale=rescale)

    @staticmethod
    def _parse_losses(losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)
        log_vars['loss'] = loss
        return loss, log_vars

    def train_step(self, data):
        losses = self.forward_train(data['img'], data['img_metas'], data['gt_bboxes'], data['gt_labels'])
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
        return outputs
