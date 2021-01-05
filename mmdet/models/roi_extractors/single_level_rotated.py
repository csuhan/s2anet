from __future__ import division

import torch

from .single_level import SingleRoIExtractor
from ..registry import ROI_EXTRACTORS


@ROI_EXTRACTORS.register_module
class SingleRoIExtractorRotated(SingleRoIExtractor):

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt((rois[:, 3] + 1) * (rois[:, 4] + 1))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def roi_rescale(self, rois, scale_factor):
        cx = rois[:, 1]
        cy = rois[:, 2]
        w = rois[:, 3] + 1
        h = rois[:, 4] + 1
        a = rois[:, 5]
        new_w = w * scale_factor
        new_h = h * scale_factor
        new_rois = torch.stack((rois[:, 0], cx, cy, new_w, new_h, a), dim=-1)
        return new_rois
