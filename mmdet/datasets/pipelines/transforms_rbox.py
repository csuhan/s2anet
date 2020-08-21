import inspect

import mmcv
import numpy as np
from numpy import random

from mmdet.core import poly2rbox, rbox2poly, rbox_flip

from ..registry import PIPELINES


@PIPELINES.register_module
class RotatedRandomFlip(object):
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability.
    """

    def __init__(self, flip_ratio=None):
        self.flip_ratio = flip_ratio
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1

    def bbox_flip(self, bboxes, img_shape):
        """Flip bboxes horizontally.

        Args:
            bboxes(ndarray): shape (..., 4*k)
            img_shape(tuple): (height, width)
        """
        assert bboxes.shape[-1] % 4 == 0
        w = img_shape[1]
        flipped = bboxes.copy()
        flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
        flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
        return flipped

    def rbox_flip(self, rboxes, img_shape):
        """Flip rboxes horizontally.

        Args:
            rboxes(ndarray): shape (..., 5*k)
            img_shape(tuple): (height, width)
        """
        assert rboxes.shape[-1] % 5 == 0
        w = img_shape[1]
        # x_ctr
        rboxes[..., 0::5] = w - rboxes[..., 0::5] - 1
        # angle
        rboxes[..., 4::5] = np.pi - rboxes[..., 4::5]

        return rboxes

    def __call__(self, results):
        if 'flip' not in results:
            flip = True if np.random.rand() < self.flip_ratio else False
            results['flip'] = flip
        if results['flip']:
            # flip image
            results['img'] = mmcv.imflip(results['img'])
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = rbox_flip(results[key],
                                         results['img_shape'])
            # flip cluster if exist
            if 'gt_clusters' in results and results['gt_clusters'] is not None:
                results['gt_clusters'] = self.bbox_flip(results['gt_clusters'],
                                                        results['img_shape'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = [mask[:, ::-1] for mask in results[key]]

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(flip_ratio={})'.format(
            self.flip_ratio)


@PIPELINES.register_module
class RotatedResize(object):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:
    - `ratio_range` is not None: randomly sample a ratio from the ratio range
        and multiply it with the image scale.
    - `ratio_range` is None and `multiscale_mode` == "range": randomly sample a
        scale from the a range.
    - `ratio_range` is None and `multiscale_mode` == "value": randomly sample a
        scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(
                results['img'], results['scale'], return_scale=True)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                results['img'], results['scale'], return_scale=True)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_rboxes(self, results):
        """
        rrect[x,y,w,h,angle] box resize
        """
        img_shape = results['img_shape']
        for key in results.get('bbox_fields', []):
            polys = rbox2poly(results[key])  # to 8 points
            polys = polys * results['scale_factor']
            # bboxes = results[key] * results['scale_factor']
            if polys.shape[0] != 0:
                polys[:, 0::2] = np.clip(polys[:, 0::2], 0, img_shape[1] - 1)
                polys[:, 1::2] = np.clip(polys[:, 1::2], 0, img_shape[0] - 1)
            rboxes = poly2rbox(polys)  # to x,y,w,h,angle
            results[key] = rboxes

    def _resize_masks(self, results):
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            if self.keep_ratio:
                masks = [
                    mmcv.imrescale(
                        mask, results['scale_factor'], interpolation='nearest')
                    for mask in results[key]
                ]
            else:
                mask_size = (results['img_shape'][1], results['img_shape'][0])
                masks = [
                    mmcv.imresize(mask, mask_size, interpolation='nearest')
                    for mask in results[key]
                ]
            results[key] = masks

    def __call__(self, results):
        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_rboxes(results)
        self._resize_masks(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(img_scale={}, multiscale_mode={}, ratio_range={}, '
                     'keep_ratio={})').format(self.img_scale,
                                              self.multiscale_mode,
                                              self.ratio_range,
                                              self.keep_ratio)
        return repr_str


@PIPELINES.register_module
class PesudoRotatedRandomFlip(object):
    def __init__(self, flip_ratio=None):
        self.flip_ratio = flip_ratio
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1

    def __call__(self, results):
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(flip_ratio={})'.format(
            self.flip_ratio)


@PIPELINES.register_module
class PesudoRotatedResize(object):
    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True):
        self.img_scale = img_scale
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    def __call__(self, results):
        results['scale_factor'] = 1.0
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(img_scale={}, multiscale_mode={}, ratio_range={}, '
                     'keep_ratio={})').format(self.img_scale,
                                              self.multiscale_mode,
                                              self.ratio_range,
                                              self.keep_ratio)
        return repr_str


@PIPELINES.register_module
class MinSizeNorm(object):
    def __init__(self, min_size=None, is_square=False):
        # min size of the edges
        self.min_size = min_size
        self.is_square = is_square

    def __call__(self, results):
        img = results['img']
        img_shape = results['img_shape']
        h, w = img_shape[:2]
        new_h = new_w = self.min_size
        if (h >= self.min_size) and (w >= self.min_size):
            if not self.is_square:
                return results
            else:
                new_h = new_w = max(h, w)
        elif (h < self.min_size) and (w >= self.min_size):
            new_w = w
        elif (h >= self.min_size) and (w < self.min_size):
            new_h = h

        img_new = np.zeros((new_h, new_w, img_shape[2]), dtype=np.uint8)
        img_new[:h, :w, :] = img
        img_shape_new = (new_h, new_w, img_shape[2])
        results['img'] = img_new
        results['img_shape'] = img_shape_new
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(min size={})'.format(
            self.min_size)
