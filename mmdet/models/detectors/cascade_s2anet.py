import torch.nn as nn

from mmdet.core import bbox2result
from .base import BaseDetector
from .. import builder
from ..registry import DETECTORS


@DETECTORS.register_module
class CascadeS2ANetDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 num_stages,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CascadeS2ANetDetector, self).__init__()
        self.num_stages = num_stages
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = nn.ModuleList()
        for head in bbox_head:
            self.bbox_head.append(builder.build_head(head))

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(CascadeS2ANetDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        for i in range(self.num_stages):
            if self.with_bbox:
                self.bbox_head[i].init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmedetection/tools/get_flops.py`
        """
        # TODO add related codes
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):

        losses = dict()

        x = self.extract_feat(img)

        featmap_sizes = [featmap.size()[-2:] for featmap in x]
        anchors_list, valid_flag_list = self.bbox_head[0].get_init_anchors(featmap_sizes, img_metas, device=x[0].device)

        for i in range(self.num_stages):
            self.current_stage = i
            lw = self.train_cfg.loss_weight[i]

            # copy anchor tensors to avoid reshape error in get_refined_anchors()
            anchors_list_cp = [
                [anchor.clone() for anchor in multi_img_anchors]
                for multi_img_anchors in anchors_list
            ]

            outs = self.bbox_head[i](x, anchors_list_cp)
            loss_inputs = outs + (
                anchors_list_cp, valid_flag_list, gt_bboxes, gt_labels, img_metas, self.train_cfg.stage_cfg[i])
            stage_loss = self.bbox_head[i].loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

            for name, value in stage_loss.items():
                mean_value = sum(value)
                losses['s{}.{}'.format(i, name)] = (
                    mean_value * lw if 'loss' in name else mean_value)

            if i < self.num_stages - 1:
                anchors_list, valid_flag_list = self.bbox_head[i].get_refine_anchors(
                    outs[1], anchors_list, featmap_sizes, img_metas, device=x[0].device)

        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)

        featmap_sizes = [featmap.size()[-2:] for featmap in x]
        anchors_list, valid_flag_list = self.bbox_head[0].get_init_anchors(featmap_sizes, img_meta,
                                                                           device=x[0].device)

        for i in range(self.num_stages):
            outs = self.bbox_head[i](x, anchors_list)
            if i < self.num_stages - 1:
                anchors_list, valid_flag_list = self.bbox_head[i].get_refine_anchors(
                    outs[1], anchors_list, featmap_sizes, img_meta, device=x[0].device)

        bbox_inputs = outs + (anchors_list, valid_flag_list, img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head[self.num_stages - 1].get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head[self.num_stages - 1].num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
