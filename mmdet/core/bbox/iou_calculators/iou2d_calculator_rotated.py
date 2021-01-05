from mmdet.ops.box_iou_rotated import box_iou_rotated
from .builder import IOU_CALCULATORS


@IOU_CALCULATORS.register_module
class BboxOverlaps2D_rotated(object):
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor): bboxes have shape (m, 5) in <x, y, w, h, a>
                format, or shape (m, 5) in <x, y, w, h, a, score> format.
            bboxes2 (Tensor): bboxes have shape (m, 5) in <x, y, w, h, a>
                format, or shape (m, 5) in <x, y, w, h, a, score> format, or be
                empty. If ``is_aligned `` is ``True``, then m and n must be
                equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 5, 6]
        assert bboxes2.size(-1) in [0, 5, 6]
        if bboxes2.size(-1) == 6:
            bboxes2 = bboxes2[..., :5]
        if bboxes1.size(-1) == 6:
            bboxes1 = bboxes1[..., :5]
        return bbox_overlaps_rotated(bboxes1, bboxes2)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str


def bbox_overlaps_rotated(rboxes1, rboxes2):
    ious = box_iou_rotated(rboxes1.float(), rboxes2.float())
    return ious
