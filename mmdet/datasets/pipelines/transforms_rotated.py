import random

import cv2
import numpy as np

from mmdet.core import poly_to_rotated_box_np, rotated_box_to_poly_np, norm_angle
from .transforms import RandomFlip, Resize
from ..registry import PIPELINES


@PIPELINES.register_module
class RotatedRandomFlip(RandomFlip):

    def bbox_flip(self, bboxes, img_shape):
        """Flip bboxes horizontally.

        Args:
            bboxes(ndarray): shape (..., 5*k)
            img_shape(tuple): (height, width)
        """
        assert bboxes.shape[-1] % 5 == 0
        w = img_shape[1]
        # x_ctr and angle
        bboxes[..., 0::5] = w - bboxes[..., 0::5] - 1
        bboxes[..., 4::5] = norm_angle(np.pi - bboxes[..., 4::5])

        return bboxes


@PIPELINES.register_module
class RotatedResize(Resize):

    def _resize_bboxes(self, results):
        img_shape = results['img_shape']
        for key in results.get('bbox_fields', []):
            polys = rotated_box_to_poly_np(results[key])  # to 8 points
            polys = polys * results['scale_factor']
            if polys.shape[0] != 0:
                polys[:, 0::2] = np.clip(polys[:, 0::2], 0, img_shape[1] - 1)
                polys[:, 1::2] = np.clip(polys[:, 1::2], 0, img_shape[0] - 1)
            rboxes = poly_to_rotated_box_np(polys)  # to x,y,w,h,angle
            results[key] = rboxes


@PIPELINES.register_module
class PesudoRotatedRandomFlip(RandomFlip):
    def __call__(self, results):
        return results


@PIPELINES.register_module
class PesudoRotatedResize(Resize):
    def __call__(self, results):
        results['scale_factor'] = 1.0
        return results


@PIPELINES.register_module
class RandomRotate(object):
    def __init__(self,
                 rate=0.5,
                 angles=[30, 60, 90, 120, 150],
                 auto_bound=False):
        self.rate = rate
        self.angles = angles
        # new image shape or not
        self.auto_bound = auto_bound

    @property
    def rand_angle(self):
        return random.sample(self.angles, 1)[0]

    @property
    def is_rotate(self):
        return np.random.rand() > self.rate

    def apply_image(self, img, bound_h, bound_w, interp=cv2.INTER_LINEAR):
        """
        img should be a numpy array, formatted as Height * Width * Nchannels
        """
        if len(img) == 0:
            return img
        return cv2.warpAffine(img, self.rm_image, (bound_w, bound_h), flags=interp)

    def apply_coords(self, coords):
        """
        coords should be a N * 2 array-like, containing N couples of (x, y) points
        """
        if len(coords) == 0:
            return coords
        coords = np.asarray(coords, dtype=float)
        return cv2.transform(coords[:, np.newaxis, :], self.rm_coords)[:, 0, :]

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=cv2.INTER_NEAREST)
        return segmentation

    def create_rotation_matrix(self, center, angle, bound_h, bound_w, offset=0):
        center = (center[0] + offset, center[1] + offset)
        rm = cv2.getRotationMatrix2D(tuple(center), angle, 1)
        if self.auto_bound:
            # Find the coordinates of the center of rotation in the new image
            # The only point for which we know the future coordinates is the center of the image
            rot_im_center = cv2.transform(
                center[None, None, :] + offset, rm)[0, 0, :]
            new_center = np.array(
                [bound_w / 2, bound_h / 2]) + offset - rot_im_center
            # shift the rotation center to the new coordinates
            rm[:, 2] += new_center
        return rm

    def filter_border(self, bboxes, h, w):
        x_ctr, y_ctr = bboxes[:, 0], bboxes[:, 1]
        keep_inds = (x_ctr > 0) & (x_ctr < w) & (y_ctr > 0) & (y_ctr < h)
        return keep_inds

    def __call__(self, results):
        # return the results directly if not rotate
        if not self.is_rotate:
            results['rotate'] = False
            return results

        h, w, c = results['img_shape']
        img = results['img']
        # angle for rotate
        angle = self.rand_angle
        results['rotate'] = True
        results['rotate_angle'] = angle

        image_center = np.array((w / 2, h / 2))
        abs_cos, abs_sin = abs(np.cos(angle)), abs(np.sin(angle))
        if self.auto_bound:
            # find the new width and height bounds
            bound_w, bound_h = np.rint(
                [h * abs_sin + w * abs_cos, h * abs_cos + w * abs_sin]
            ).astype(int)
        else:
            bound_w, bound_h = w, h

        self.rm_coords = self.create_rotation_matrix(
            image_center, angle, bound_h, bound_w)
        # Needed because of this problem https://github.com/opencv/opencv/issues/11784
        self.rm_image = self.create_rotation_matrix(
            image_center, angle, bound_h, bound_w, offset=-0.5)
        # rotate img
        img = self.apply_image(img, bound_h, bound_w)
        results['img'] = img
        results['img_shape'] = (bound_h, bound_w, c)
        # rotate bboxes
        gt_bboxes = results.get('gt_bboxes', [])
        labels = results.get('gt_labels', [])

        polys = rotated_box_to_poly_np(gt_bboxes).reshape(-1, 2)
        polys = self.apply_coords(polys).reshape(-1, 8)
        gt_bboxes = poly_to_rotated_box_np(polys)
        keep_inds = self.filter_border(gt_bboxes, bound_h, bound_w)
        gt_bboxes = gt_bboxes[keep_inds, :]
        labels = labels[keep_inds]
        if len(gt_bboxes) == 0:
            return None
        results['gt_bboxes'] = gt_bboxes
        results['gt_labels'] = labels
        return results
