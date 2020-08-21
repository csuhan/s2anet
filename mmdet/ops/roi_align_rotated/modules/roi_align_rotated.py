from torch.nn.modules.module import Module
from torch import nn
from ..functions.roi_align_rotated import RoIAlignRotatedFunction
from ..functions.roi_align_rotated import roi_align_rotated


class RoIAlignRotated(Module):

    def __init__(self, out_size, spatial_scale, sample_num=0):
        super(RoIAlignRotated, self).__init__()

        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)

    def forward(self, features, rois):
        return RoIAlignRotatedFunction.apply(features, rois, self.out_size,
                                      self.spatial_scale, self.sample_num)




class ModulatedRoIAlignRotatedPack(RoIAlignRotated):

    def __init__(self,
                 out_size,
                 spatial_scale,
                 sample_num,
                 num_mask_fcs=2,
                 out_channels=256,
                 fc_channels=1024):
        super(ModulatedRoIAlignRotatedPack, self).__init__(
            out_size, spatial_scale, sample_num)
        # TODO: carefully set the channels

        self.out_channels = out_channels
        self.num_mask_fcs = num_mask_fcs
        self.fc_channels = fc_channels

        mask_fc_seq = []
        ic = self.out_size * self.out_size * self.out_channels
        for i in range(self.num_mask_fcs):
            if i < self.num_mask_fcs - 1:
                oc = self.fc_channels
            else:
                oc = self.out_size * self.out_size
            mask_fc_seq.append(nn.Linear(ic, oc))
            ic = oc
            if i < self.num_mask_fcs - 1:
                mask_fc_seq.append(nn.ReLU(inplace=True))
            else:
                mask_fc_seq.append(nn.Sigmoid())
        self.mask_fc = nn.Sequential(*mask_fc_seq)
        # TODO: check out the -2
        # import pdb
        # pdb.set_trace()
        self.mask_fc[-2].weight.data.zero_()
        self.mask_fc[-2].bias.data.zero_()

    def forward(self, data, rois):
        assert data.size(1) == self.out_channels

        n = rois.shape[0]

        x= roi_align_rotated(data, rois, self.out_size,
                                      self.spatial_scale, self.sample_num)
        mask = self.mask_fc(x.view(n, -1))
        mask = mask.view(n, 1, self.out_size, self.out_size)
        return x * mask



