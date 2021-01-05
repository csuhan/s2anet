import torch


class AnchorGeneratorRotated(object):
    def __init__(self, base_size, scales, ratios, angles=[0,],scale_major=True, ctr=None):
        self.base_size = base_size
        self.scales = torch.Tensor(scales)
        self.ratios = torch.Tensor(ratios)
        self.angles = torch.Tensor(angles)
        self.scale_major = scale_major
        self.ctr = ctr
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        return self.base_anchors.size(0)

    def gen_base_anchors(self):
        w = self.base_size
        h = self.base_size
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = self.ctr

        h_ratios = torch.sqrt(self.ratios)
        w_ratios = 1 / h_ratios
        assert self.scale_major, "AnchorGeneratorRotated only support scale-major anchors!"

        ws = (w * w_ratios[:, None, None] * self.scales[None, :, None] *
              torch.ones_like(self.angles)[None, None, :]).view(-1)
        hs = (h * h_ratios[:, None, None] * self.scales[None, :, None] *
              torch.ones_like(self.angles)[None, None, :]).view(-1)
        angles = self.angles.repeat(len(self.scales) * len(self.ratios))

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        x_ctr += torch.zeros_like(ws)
        y_ctr += torch.zeros_like(ws)
        base_anchors = torch.stack(
            [x_ctr, y_ctr, ws, hs, angles], dim=-1)

        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_size, stride=16, device='cuda'):
        # featmap_size*stride project it to original area
        base_anchors = self.base_anchors.to(device)

        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shift_others = torch.zeros_like(shift_xx)
        shifts = torch.stack(
            [shift_xx, shift_yy, shift_others, shift_others, shift_others], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 5) to K shifts (K, 1, 5) to get
        # shifted anchors (K, A, 5), reshape to (K*A, 5)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 5)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(
            valid.size(0), self.num_base_anchors).contiguous().view(-1)
        return valid
