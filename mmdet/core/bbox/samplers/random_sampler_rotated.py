import torch

from .random_sampler import RandomSampler
from .sampling_result import SamplingResult
from ..builder import BBOX_SAMPLERS


@BBOX_SAMPLERS.register_module
class RandomSamplerRotated(RandomSampler):

    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               **kwargs):

        gt_bboxes = gt_bboxes.float()
        bboxes = bboxes.float()

        if len(bboxes.shape) < 2:
            bboxes = bboxes[None, :]
        # this is the only difference between RandomSamplerRotated and RandomSampler
        bboxes = bboxes[:, :5]

        gt_flags = bboxes.new_zeros((bboxes.shape[0],), dtype=torch.uint8)
        if self.add_gt_as_proposals:
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(
            assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos

        # print('Pos:{} Neg:{}'.format(num_sampled_pos,num_expected_neg))

        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(
            assign_result, num_expected_neg, bboxes=bboxes, **kwargs)
        neg_inds = neg_inds.unique()

        return SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                              assign_result, gt_flags)
