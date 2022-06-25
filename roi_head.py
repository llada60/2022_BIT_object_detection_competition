import numpy as np
import torch
from mmcv.runner import ModuleList
from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms)
from .builder import (build_roi_extractor,build_head)
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin

class RoIHead(BaseRoIHead,BBoxTestMixin):
    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RoIHead, self).__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg
        )
        self.num_stages=num_stages
        self.stage_loss_weights=stage_loss_weights

    def init_bbox_head(self,bbox_roi_extractor,bbox_head):
        # bbox_roi_extract(dict): box roi extractor的config
        # bbox_head(dict): box head的config

        self.bbox_roi_extractor=ModuleList()
        self.bbox_head=ModuleList()
        # 保证bbox_roi_extractor和bbox_head为list类型
        if not isinstance(bbox_roi_extractor,list):
            bbox_roi_extractor=[
                bbox_roi_extractor for _ in range(self.num_stages)
            ]
        if not isinstance(bbox_head,list):
            bbox_head=[
                bbox_head for _ in range(self.num_stages)
            ]
        assert  len(bbox_roi_extractor)==len(bbox_head)==self.num_stages
        for roi_extractor,head in zip(bbox_roi_extractor,bbox_head):
            self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
            self.bbox_head.append(build_head(head))

    def init_assigner_sampler(self):
        self.bbox_assigner=[] # 正负样本属性分配：进行正负样本（前景/背景）分配
        self.bbox_sampler=[] # 正负样本采样控制：控制正负样本数量比
        if self.train_cfg is not None:
            for idx,rcnn_train_cfg in enumerate(self.train_cfg):
                self.bbox_assigner.append(
                    build_assigner(rcnn_train_cfg.assigner)
                )
                self.current_stage=idx
                self.bbox_sampler.append(
                    build_sampler(rcnn_train_cfg.sampler,context=self)
                )

    def forward_dummy(self,x,proposals):
        outs=()
        rois=bbox2roi([proposals])
        for i in range(self.num_stages):
            bbox_results=self._bbox_forward(i,x,rois)
            outs=outs+(bbox_results['cls_score'],
                       bbox_results['bbox_pred'])
        return outs

    def _bbox_forward(self,stage,x,rois):
        bbox_roi_extractor=self.bbox_roi_extractor[stage]
        bbox_head=self.bbox_head[stage]
        bbox_feats=bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],rois)
        cls_score,bbox_pred=bbox_head(bbox_feats)
        bbox_results=dict(
            cls_score=cls_score,bbox_pred=bbox_pred,bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self,stage,x,sampling_results,gt_bboxes,
                            gt_labels,rcnn_train_cfg):
        rois=bbox2roi([res.bboxes for res in sampling_results])
        bbox_results=self._bbox_forward(stage,x,rois)
        bbox_targets=self.bbox_head[stage].get_targets(
            sampling_results,gt_bboxes,gt_labels,rcnn_train_cfg
        )

        loss_bbox=self.bbox_head[stage].loss(bbox_results['cls_score'],
                                             bbox_results['bbox_pred'],rois,
                                             *bbox_targets)

        bbox_results.update(
            loss_bbox=loss_bbox,rois=rois,bbox_targets=bbox_targets
        )
        return bbox_results

    def forward_train(self,
                      x, # x(list[tensor]) 图像的多层特征
                      img_metas, # img_metas(list[dict]) 图像信息：img_shape,scale_factor,flip...
                      proposal_list, # region proposal
                      gt_bboxes, # ground truth bboxes: shape(num_gts,4) [tl_x,tl_y,br_x,br_y]
                      gt_labels, # 标签
                      gt_bboxes_ignore=None): # 计算loss时 bboxes可以忽略

        losses=dict()
        for i in range(self.num_stages):
            self.current_stage=i
            rcnn_train_cfg=self.train_cfg[i]
            lw=self.stage_loss_weights[i]

            # 给ground_truth 和 采样的proposals 进行赋值
            sampling_results=[]
            bbox_assigner=self.bbox_assigner[i]
            bbox_sampler=self.bbox_sampler[i]
            num_imgs=len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore=[None for _ in range(num_imgs)]

            for j in range(num_imgs):
                assign_result=bbox_assigner.assign(
                    proposal_list[j],gt_bboxes[j],gt_bboxes_ignore[j],gt_labels[j]
                )
                sampling_result=bbox_sampler.sample(
                    assign_result,
                    proposal_list[j],
                    gt_bboxes[j],
                    gt_labels[j],
                    feats=[level_feat[j][None] for level_feat in x]
                )
                sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_results = self._bbox_forward_train(i,x,sampling_results,
                                                    gt_bboxes,gt_labels,
                                                    rcnn_train_cfg)
            for name,value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}']=(
                    value*lw if 'loss' in name else value
                )

            # 优化bboxes
            if i<self.num_stages-1:
                pos_is_gts=[res.pos_is_gt for res in sampling_results]
                roi_labels=bbox_results['bbox_targets'][0]
                # 分类
                with torch.no_grad():
                    cls_score=bbox_results['cls_score']
                    if self.bbox_head[i].custom_activation:
                        cls_score=self.bbox_head.loss_cls.get_activation(cls_score)

                    # 空的proposal
                    if cls_score.numel()==0:
                        break

                    # roi_labels==self.bbox_head[i].num_classes?cls_score[:,:-1].argmax(1):roi_labels
                    roi_labels=torch.where(
                        roi_labels==self.bbox_head[i].num_classes,
                        cls_score[:,:-1].argmax(1),roi_labels
                    )
                    proposal_list=self.bbox_head[i].refine_bboxes(
                        bbox_results['rois'],roi_labels,
                        bbox_results['bbox_pred'],pos_is_gts,img_metas
                    )
            return losses

    def simple_test(self,x,proposal_list,img_metas,rescale=False):
        '''
            x:上游网络的特征 (batch_size,c,h,x)
            proposal_list:rpn_head生成的proposals (num_proposals,(x1,y1,x2,y2,score))
            img_metas: 图片变换情况
            rescale: 默认为真，是否将图像规范成原始图像大小
        '''

        assert  self.with_bbox,'Bbox head must be implemented.'
        num_imgs=len(proposal_list)
        img_shapes=tuple(meta['img_shape']for meta in img_metas)
        ori_shapes=tuple(meta['ori_shape']for meta in img_metas)
        scale_factors=tuple(meta['scale_factor']for meta in img_metas)

        # multi-stage
        ms_bbox_result={}
        ms_segm_result={}
        ms_scores=[]
        rcnn_test_cfg=self.test_cfg

        rois=bbox2roi(proposal_list)

        if rois.shape[0]==0:
            bbox_results=[[
                np.zeros((0,5),dtype=np.float32)
                for _ in range(self.bbox_head[-1].num_classes)
            ]]*num_imgs

            results=bbox_results

            return results

        for i in range(self.num_stages):
            bbox_results=self._bbox_forward(i,x,rois)
            cls_score=bbox_results['cls_score']
            bbox_pred=bbox_results['bbox_pred']
            num_proposals_per_img=tuple(
                len(proposals) for proposals in proposal_list
            )
            rois=rois.split(num_proposals_per_img,0)
            cls_score=cls_score.split(num_proposals_per_img,0)
            if isinstance(bbox_pred,torch.Tensor):
                bbox_pred=bbox_pred.split(num_proposals_per_img,0)
            else:
                bbox_pred=self.bbox_head[i].bbox_pred_split(
                    bbox_pred,num_proposals_per_img
                )
            ms_scores.append(cls_score)

            if i<self.num_stages-1:
                if self.bbox_head[i].custom_activation:
                    cls_score=[
                        self.bbox_head[i].loss_cls.get_activation(s)
                        for s in cls_score
                    ]
                refine_rois_list=[]
                for j in range(num_imgs):
                    if rois[j].shape[0]>0:
                        bbox_label = cls_score[j][:,:-1].argmax(dim=1)
                        refined_rois=self.bbox_head[i].regress_by_class(
                            rois[j],bbox_label,bbox_pred[j],img_metas[j]
                        )
                        refine_rois_list.append(refined_rois)
                rois=torch.cat(refine_rois_list)

        # 每个阶段每张图片的类别分数
        cls_score=[
            sum([score[i] for score in ms_scores])/float(len(ms_scores))
            for i in range(num_imgs)
        ]

        # bbox后向传播
        det_bboxes=[]
        det_labels=[]
        for i in range(num_imgs):
            det_bbox,det_label=self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg
            )
            det_bbox.append(det_bbox)
            det_labels.append(det_label)

        bbox_results=[
            bbox2result(det_bboxes[i],det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble']=bbox_results

    def aug_test(self,features,proposal_list,img_metas,rescale=False):
        # rescale=False, return bboxes会是imgs[0]大小
        rcnn_test_cfg=self.test_cfg
        aug_bboxes=[]
        aug_scores=[]
        for x,img_meta in zip(features,img_metas):
            img_shape=img_meta[0]['img_shape']
            scale_factor=img_meta[0]['scale_factor']
            flip=img_meta[0]['flip']
            flip_direction=img_meta[0]['flip_direction']

            proposals=bbox_mapping(proposal_list[0][:,:4],img_shape,
                                   scale_factor,flip,flip_direction)
            ms_scores=[]

            rois=bbox2roi([proposals])

            if rois.shape[0]==0:
                # 图片中没有proposal
                aug_bboxes.append(rois.new_zeros(0,4))
                aug_scores.append(rois.new_zeros(0,1))
                continue

            for i in range(self.num_stages):
                bbox_results=self._bbox_forward(i,x,rois)
                ms_scores.append(bbox_results['cls_score'])

                if i<self.num_stages-1:
                    cls_score=bbox_results['cls_score']
                    if self.bbox_head[i].custom_activation:
                        cls_score=self.bbox_head[i].loss_cls.get_activation(
                            cls_score
                        )
                        bbox_label=cls_score[:,:-1].argmax(dim=1)
                        rois=self.bbox_head[i].regress_by_class(
                            rois,bbox_label,bbox_results['bbox_pred'],
                            img_meta[0]
                        )
            cls_score=sum(ms_scores)/float(len(ms_scores))
            bboxes,scores=self.bbox_head[-1].get_bboxes(
                rois,
                cls_score,
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None
            )
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

            # 合并后，bboxes会变换成原始图像大小
            merged_bboxes,merged_scores=merge_aug_bboxes(
                aug_bboxes,aug_scores,img_metas,rcnn_test_cfg
            )
            det_bboxes,det_labels=multiclass_nms(merged_bboxes,merged_scores,
                                                 rcnn_test_cfg.score_thr,
                                                 rcnn_test_cfg.nms,
                                                 rcnn_test_cfg.max_per_img)
            bbox_result=bbox2result(det_bboxes,det_labels,
                                    self.bbox_head[-1].num_classes)

    def onnx_export(self,x,proposals,img_metas):
        assert self.with_bbox,'Bbox head must be implemented.'
        assert proposals.shape[0]==1,'Only support one input image '\
                                    'while in exporting to ONNX'
        rois=proposals[...,:-1]
        batch_size=rois.shape[0]
        num_proposals_per_img=rois.shape[1]
        rois=rois.view(-1,4)

        rois=torch.cat([rois.new_zeros(rois.shape[0],1),rois],dim=-1)

        max_shape=img_metas[0]['img_shape_for_onnx']
        ms_scores=[]
        rcnn_test_cfg=self.test_cfg

        for i in range(self.num_stages):
            bbox_results=self._bbox_forward(i,x,rois)

            cls_score=bbox_results['cls_score']
            bbox_pred=bbox_results['bbox_pred']
            rois=rois.reshape(batch_size,num_proposals_per_img,
                              rois.size(-1))
            cls_score=cls_score.reshape(batch_size,num_proposals_per_img,
                                        cls_score.size(-1))
            bbox_pred=bbox_pred.reshape(batch_size,num_proposals_per_img,4)
            ms_scores.append(cls_score)
            if i<self.num_stages-1:
                assert self.bbox_head[i].reg_class_agnostic
                new_rois=self.bbox_head[i].bbox_coder.decode(
                    rois[...,1:],bbox_pred,max_shape=max_shape
                )
                rois = new_rois.reshape(-1,new_rois.shape[-1])
                rois=torch.cat([rois.new_zeros(rois.shape[0],1),rois],dim=-1)

        cls_score=sum(ms_scores)/float(len(ms_scores))
        bbox_pred=bbox_pred.reshape(batch_size, num_proposals_per_img,4)
        rois=rois.reshape(batch_size,num_proposals_per_img,-1)
        det_bboxes,det_labels=self.bbox_head[-1].onnx_export(
            rois,cls_score,bbox_pred,max_shape,cfg=rcnn_test_cfg
        )
        return det_bboxes,det_labels
