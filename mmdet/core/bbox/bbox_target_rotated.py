import torch

from .transforms_rotated import bbox2delta_rotated
from ..utils import multi_apply


def bbox_target_rotated(pos_bboxes_list,
                        neg_bboxes_list,
                        pos_gt_bboxes_list,
                        pos_gt_labels_list,
                        cfg,
                        reg_classes=1,
                        target_means=[.0, .0, .0, .0, .0],
                        target_stds=[1.0, 1.0, 1.0, 1.0, 1.0],
                        concat=True):
    labels, label_weights, bbox_targets, bbox_weights = multi_apply(
        bbox_target_single,
        pos_bboxes_list,
        neg_bboxes_list,
        pos_gt_bboxes_list,
        pos_gt_labels_list,
        cfg=cfg,
        reg_classes=reg_classes,
        target_means=target_means,
        target_stds=target_stds)

    if concat:
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)
    return labels, label_weights, bbox_targets, bbox_weights


def bbox_target_single(pos_bboxes,
                       neg_bboxes,
                       pos_gt_bboxes,
                       pos_gt_labels,
                       cfg,
                       reg_classes=1,
                       target_means=[.0, .0, .0, .0, .0],
                       target_stds=[1.0, 1.0, 1.0, 1.0, 1.0]):
    num_pos = pos_bboxes.size(0)
    num_neg = neg_bboxes.size(0)
    num_samples = num_pos + num_neg
    labels = pos_bboxes.new_zeros(num_samples, dtype=torch.long)
    label_weights = pos_bboxes.new_zeros(num_samples)
    bbox_targets = pos_bboxes.new_zeros(num_samples, 5)
    bbox_weights = pos_bboxes.new_zeros(num_samples, 5)
    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
        pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
        label_weights[:num_pos] = pos_weight
        pos_bbox_targets = bbox2delta_rotated(pos_bboxes, pos_gt_bboxes, target_means,
                                              target_stds)
        bbox_targets[:num_pos, :] = pos_bbox_targets
        bbox_weights[:num_pos, :] = 1
    if num_neg > 0:
        label_weights[-num_neg:] = 1.0
    return labels, label_weights, bbox_targets, bbox_weights
