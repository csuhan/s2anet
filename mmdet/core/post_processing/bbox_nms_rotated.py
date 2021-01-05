import torch

from mmdet.ops import ml_nms_rotated


def multiclass_nms_rotated(multi_bboxes,
                           multi_scores,
                           score_thr,
                           nms_cfg,
                           max_num=-1,
                           score_factors=None):
    """NMS for multi-class bboxes.
    Args:
        multi_bboxes (Tensor): shape (n, #class*5) or (n, 5)
        multi_scores (Tensor): shape (n, #class), where the 0th column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS
    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 6) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 5:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 5)[:, 1:]
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 5)
    scores = multi_scores[:, 1:]

    # filter out boxes with low scores
    valid_mask = scores > score_thr
    bboxes = bboxes[valid_mask]
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]
    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 6))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)
        return bboxes, labels
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    iou_thr = nms_cfg_.pop('iou_thr', 0.1)
    labels = labels.to(bboxes)
    keep = ml_nms_rotated(bboxes, scores, labels, iou_thr)
    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if keep.size(0) > max_num:
        _, inds = scores.sort(descending=True)
        inds = inds[:max_num]
        bboxes = bboxes[inds]
        scores = scores[inds]
        labels = labels[inds]

    return torch.cat([bboxes, scores[:, None]], 1), labels
