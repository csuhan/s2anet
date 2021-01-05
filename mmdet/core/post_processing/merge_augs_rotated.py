import torch

from mmdet.ops import nms_rotated
from ..bbox import bbox_mapping_back_rotated


def merge_aug_proposals_rotated(aug_proposals, img_metas, rpn_test_cfg):
    """Merge augmented proposals (multiscale, flip, etc.)

    Args:
        aug_proposals (list[Tensor]): proposals from different testing
            schemes, shape (n, 6). Note that they are not rescaled to the
            original image size.

        img_metas (list[dict]): list of image info dict where each dict has:
            'img_shape', 'scale_factor', 'flip', and my also contain
            'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            For details on the values of these keys see
            `mmdet/datasets/pipelines/formatting.py:Collect`.

        rpn_test_cfg (dict): rpn test config.

    Returns:
        Tensor: shape (n, 5), proposals corresponding to original image scale.
    """
    recovered_proposals = []
    for proposals, img_info in zip(aug_proposals, img_metas):
        img_shape = img_info['img_shape']
        scale_factor = img_info['scale_factor']
        flip = img_info['flip']
        _proposals = proposals.clone()
        _proposals[:, :5] = bbox_mapping_back_rotated(_proposals[:, :5], img_shape,
                                                      scale_factor, flip)
        recovered_proposals.append(_proposals)
    aug_proposals = torch.cat(recovered_proposals, dim=0)
    merged_proposals, _ = nms_rotated(aug_proposals, rpn_test_cfg.nms_thr)
    scores = merged_proposals[:, 5]
    _, order = scores.sort(0, descending=True)
    num = min(rpn_test_cfg.max_num, merged_proposals.shape[0])
    order = order[:num]
    merged_proposals = merged_proposals[order, :]
    return merged_proposals


def merge_aug_bboxes_rotated(aug_bboxes, aug_scores, img_metas, rcnn_test_cfg):
    """Merge augmented detection bboxes and scores.

    Args:
        aug_bboxes (list[Tensor]): shape (n, 5*#class)
        aug_scores (list[Tensor] or None): shape (n, #class)
        img_shapes (list[Tensor]): shape (3, ).
        rcnn_test_cfg (dict): rcnn test config.

    Returns:
        tuple: (bboxes, scores)
    """
    recovered_bboxes = []
    for bboxes, img_info in zip(aug_bboxes, img_metas):
        img_shape = img_info[0]['img_shape']
        scale_factor = img_info[0]['scale_factor']
        flip = img_info[0]['flip']
        bboxes = bbox_mapping_back_rotated(bboxes, img_shape, scale_factor, flip)
        recovered_bboxes.append(bboxes)
    bboxes = torch.stack(recovered_bboxes).mean(dim=0)
    if aug_scores is None:
        return bboxes
    else:
        scores = torch.stack(aug_scores).mean(dim=0)
        return bboxes, scores
