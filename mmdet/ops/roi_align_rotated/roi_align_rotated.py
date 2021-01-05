# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from .roi_align_rotated_cuda import roi_align_rotated_forward, roi_align_rotated_backward


class _ROIAlignRotated(Function):
    @staticmethod
    def forward(ctx, input, roi, out_size, spatial_scale, sampling_ratio):
        ctx.save_for_backward(roi)
        ctx.out_size = out_size
        ctx.spatial_scale = spatial_scale
        ctx.sample_num = sampling_ratio
        ctx.input_shape = input.size()
        output = roi_align_rotated_forward(
            input, roi, spatial_scale, out_size[0], out_size[1], sampling_ratio
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (rois,) = ctx.saved_tensors
        output_size = ctx.out_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sample_num
        bs, ch, h, w = ctx.input_shape
        grad_input = roi_align_rotated_backward(
            grad_output,
            rois,
            spatial_scale,
            output_size[0],
            output_size[1],
            bs,
            ch,
            h,
            w,
            sampling_ratio,
        )
        return grad_input, None, None, None, None, None


roi_align_rotated = _ROIAlignRotated.apply


class RoIAlignRotated(nn.Module):
    def __init__(self, out_size, spatial_scale, sample_num):
        """
        Args:
            out_size (tuple): h, w
            spatial_scale (float): scale the input boxes by this number
            sample_num (int): number of inputs samples to take for each output
                sample. 0 to take samples densely.

        Note:
            roi_align_rotated supports continuous coordinate by default:
            Given a continuous coordinate c, its two neighboring pixel indices (in our
            pixel model) are computed by floor(c - 0.5) and ceil(c - 0.5). For example,
            c=1.3 has pixel neighbors with discrete indices [0] and [1] (which are sampled
            from the underlying signal at continuous coordinates 0.5 and 1.5).
        """
        super(RoIAlignRotated, self).__init__()
        self.out_size = _pair(out_size)
        self.spatial_scale = spatial_scale
        self.sample_num = sample_num

    def forward(self, input, rois):
        """
        Args:
            input: NCHW images
            rois: Bx6 boxes. First column is the index into N.
                The other 5 columns are (x_ctr, y_ctr, width, height, angle_degrees).
        """
        assert rois.dim() == 2 and rois.size(1) == 6
        orig_dtype = input.dtype
        if orig_dtype == torch.float16:
            input = input.float()
            rois = rois.float()
        return roi_align_rotated(
            input, rois, self.out_size, self.spatial_scale, self.sample_num
        ).to(dtype=orig_dtype)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "out_size=" + str(self.out_size[0])
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sample_num=" + str(self.sample_num)
        tmpstr += ")"
        return tmpstr
