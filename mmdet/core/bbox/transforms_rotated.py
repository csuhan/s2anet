import math

import numpy as np
import torch


def norm_angle(angle, range=[-np.pi / 4, np.pi]):
    return (angle - range[0]) % range[1] + range[0]


def bbox2delta_rotated(proposals, gt, means=(0., 0., 0., 0., 0.), stds=(1., 1., 1., 1., 1.)):
    """Compute deltas of proposals w.r.t. gt.

    We usually compute the deltas of x, y, w, h of proposals w.r.t ground
    truth bboxes to get regression target.
    This is the inverse function of :func:`delta2bbox`.

    Args:
        proposals (Tensor): Boxes to be transformed, shape (N, ..., 5)
        gt (Tensor): Gt bboxes to be used as base, shape (N, ..., 5)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates

    Returns:
        Tensor: deltas with shape (N, 5), where columns represent dx, dy,
            dw, dh.
    """
    assert proposals.size() == gt.size()

    gt_widths = gt[..., 2]
    gt_heights = gt[..., 3]
    gt_angle = gt[..., 4]

    proposals_widths = proposals[..., 2]
    proposals_heights = proposals[..., 3]
    proposals_angles = proposals[..., 4]

    cosa = torch.cos(proposals_angles)
    sina = torch.sin(proposals_angles)
    coord = gt[..., 0:2] - proposals[..., 0:2]

    dx = (cosa * coord[..., 0] + sina * coord[..., 1]) / proposals_widths
    dy = (-sina * coord[..., 0] + cosa * coord[..., 1]) / proposals_heights
    dw = torch.log(gt_widths / proposals_widths)
    dh = torch.log(gt_heights / proposals_heights)
    da = (gt_angle - proposals_angles)
    da = norm_angle(da) / np.pi

    deltas = torch.stack((dx, dy, dw, dh, da), -1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2bbox_rotated(rois, deltas, means=(0., 0., 0., 0., 0.), stds=(1., 1., 1., 1., 1.), max_shape=None,
                       wh_ratio_clip=16 / 1000, clip_border=True):
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of :func:`bbox2delta`.

    Args:
        rois (Tensor): Boxes to be transformed. Has shape (N, 5)
        deltas (Tensor): Encoded offsets with respect to each roi.
            Has shape (N, 5 * num_classes). Note N = num_anchors * W * H when
            rois is a grid of anchors. Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates
        max_shape (tuple[int, int]): Maximum bounds for boxes. specifies (H, W)
        wh_ratio_clip (float): Maximum aspect ratio for boxes.
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.

    Returns:
        Tensor: Boxes with shape (N, 5), where columns represent

    References:
        .. [1] https://arxiv.org/abs/1311.2524
    """
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means

    dx = denorm_deltas[:, 0::5]
    dy = denorm_deltas[:, 1::5]
    dw = denorm_deltas[:, 2::5]
    dh = denorm_deltas[:, 3::5]
    dangle = denorm_deltas[:, 4::5]

    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    roi_x = (rois[:, 0]).unsqueeze(1).expand_as(dx)
    roi_y = (rois[:, 1]).unsqueeze(1).expand_as(dy)
    roi_w = (rois[:, 2]).unsqueeze(1).expand_as(dw)
    roi_h = (rois[:, 3]).unsqueeze(1).expand_as(dh)
    roi_angle = (rois[:, 4]).unsqueeze(1).expand_as(dangle)
    gx = dx * roi_w * torch.cos(roi_angle) \
         - dy * roi_h * torch.sin(roi_angle) + roi_x
    gy = dx * roi_w * torch.sin(roi_angle) \
         + dy * roi_h * torch.cos(roi_angle) + roi_y
    gw = roi_w * dw.exp()
    gh = roi_h * dh.exp()

    ga = np.pi * dangle + roi_angle
    ga = norm_angle(ga)

    bboxes = torch.stack([gx, gy, gw, gh, ga], dim=-1).view_as(deltas)
    return bboxes


def bbox_flip_rotated(bboxes, img_shape):
    """
    Flip bboxes horizontally
    :param bboxes: (Tensor): Shape (..., 5*k), (x_ctr, y_ctr, w, h, angle)
    :param img_shape: (tuple): Image shape.
    :return: Same type as 'dbboxes': Flipped dbboxes
    """
    assert bboxes.shape[-1] % 5 == 0
    if isinstance(bboxes, torch.Tensor):
        flipped = bboxes.clone()
        # flip x
        flipped[:, 0::5] = img_shape[1] - bboxes[:, 0::5] - 1
        # flip angle
        angle = -bboxes[:, 4::5]
        angle = norm_angle(angle)
        flipped[:, 4::5] = angle
        return flipped
    elif isinstance(bboxes, np.ndarray):
        flipped = bboxes.copy()
        # flip x
        flipped[..., 0::5] = img_shape[1] - bboxes[..., 0::5] - 1
        # flip angle
        angle = -bboxes[..., 4::5]
        angle = norm_angle(angle)
        flipped[..., 4::5] = angle
        return flipped


def bbox_mapping_rotated(dbboxes, img_shape, scale_factor, flip):
    """
    Map dbboxes from testing scale to original image scale
    :param dbboxes:
    :param img_shape:
    :param scale_factor:
    :param flip:
    :return:
    """
    new_dbboxes = dbboxes.clone()
    new_dbboxes[..., 0::5] = dbboxes[..., 0::5] * scale_factor
    new_dbboxes[..., 1::5] = dbboxes[..., 1::5] * scale_factor
    new_dbboxes[..., 2::5] = dbboxes[..., 2::5] * scale_factor
    new_dbboxes[..., 3::5] = dbboxes[..., 3::5] * scale_factor
    if flip:
        new_dbboxes = bbox_flip_rotated(new_dbboxes, img_shape)

    return new_dbboxes


# Test passed


def bbox_mapping_back_rotated(dbboxes, img_shape, scale_factor, flip):
    """
    Map dbboxes from testing scael to original image scale
    :param dbboxes:
    :param img_shape:
    :param scale_factor:
    :param flip:
    :return:
    """
    new_dbboxes = bbox_flip_rotated(dbboxes, img_shape) if flip else dbboxes
    new_dbboxes[..., 0::5] = new_dbboxes[..., 0::5] / scale_factor
    new_dbboxes[..., 1::5] = new_dbboxes[..., 1::5] / scale_factor
    new_dbboxes[..., 2::5] = new_dbboxes[..., 2::5] / scale_factor
    new_dbboxes[..., 3::5] = new_dbboxes[..., 3::5] / scale_factor
    return new_dbboxes


def bbox2result_rotated(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 6) 0-4:bbox 5:score
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [
            np.zeros((0, 6), dtype=np.float32) for i in range(num_classes - 1)
        ]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes - 1)]


def rotated_box_to_poly_single(rrect):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    x_ctr, y_ctr, width, height, angle = rrect[:5]
    tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
    rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    poly = R.dot(rect)
    x0, x1, x2, x3 = poly[0, :4] + x_ctr
    y0, y1, y2, y3 = poly[1, :4] + y_ctr
    poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3], dtype=np.float32)
    poly = get_best_begin_point_single(poly)
    return poly


def rotated_box_to_poly_np(rrects):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    polys = []
    for rrect in rrects:
        x_ctr, y_ctr, width, height, angle = rrect[:5]
        tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
        rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        poly = R.dot(rect)
        x0, x1, x2, x3 = poly[0, :4] + x_ctr
        y0, y1, y2, y3 = poly[1, :4] + y_ctr
        poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3], dtype=np.float32)
        polys.append(poly)
    polys = np.array(polys)
    polys = get_best_begin_point(polys)
    return polys


def rotated_box_to_poly(rboxes):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    N = rboxes.shape[0]
    x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
        1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
    tl_x, tl_y, br_x, br_y = -width * 0.5, -height * 0.5, width * 0.5, height * 0.5

    rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y,
                         br_y, br_y], dim=0).reshape(2, 4, N).permute(2, 0, 1)

    sin, cos = torch.sin(angle), torch.cos(angle)
    # M.shape=[N,2,2]
    M = torch.stack([cos, -sin, sin, cos],
                    dim=0).reshape(2, 2, N).permute(2, 0, 1)
    # polys:[N,8]
    polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
    polys[:, ::2] += x_ctr.unsqueeze(1)
    polys[:, 1::2] += y_ctr.unsqueeze(1)

    return polys


def poly_to_rotated_box_single(poly):
    """
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    to
    rotated_box:[x_ctr,y_ctr,w,h,angle]
    """
    poly = np.array(poly[:8], dtype=np.float32)

    pt1 = (poly[0], poly[1])
    pt2 = (poly[2], poly[3])
    pt3 = (poly[4], poly[5])
    pt4 = (poly[6], poly[7])

    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) +
                    (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) +
                    (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

    width = max(edge1, edge2)
    height = min(edge1, edge2)

    angle = 0
    if edge1 > edge2:
        angle = np.arctan2(
            np.float(pt2[1] - pt1[1]), np.float(pt2[0] - pt1[0]))
    elif edge2 >= edge1:
        angle = np.arctan2(
            np.float(pt4[1] - pt1[1]), np.float(pt4[0] - pt1[0]))

    angle = norm_angle(angle)

    x_ctr = np.float(pt1[0] + pt3[0]) / 2
    y_ctr = np.float(pt1[1] + pt3[1]) / 2
    rotated_box = np.array([x_ctr, y_ctr, width, height, angle])
    return rotated_box


def poly_to_rotated_box_np(polys):
    """
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    to
    rotated_boxes:[x_ctr,y_ctr,w,h,angle]
    """
    rotated_boxes = []
    for poly in polys:
        poly = np.array(poly[:8], dtype=np.float32)

        pt1 = (poly[0], poly[1])
        pt2 = (poly[2], poly[3])
        pt3 = (poly[4], poly[5])
        pt4 = (poly[6], poly[7])

        edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) +
                        (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
        edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) +
                        (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

        width = max(edge1, edge2)
        height = min(edge1, edge2)

        angle = 0
        if edge1 > edge2:
            angle = np.arctan2(
                np.float(pt2[1] - pt1[1]), np.float(pt2[0] - pt1[0]))
        elif edge2 >= edge1:
            angle = np.arctan2(
                np.float(pt4[1] - pt1[1]), np.float(pt4[0] - pt1[0]))

        angle = norm_angle(angle)

        x_ctr = np.float(pt1[0] + pt3[0]) / 2
        y_ctr = np.float(pt1[1] + pt3[1]) / 2
        rotated_box = np.array([x_ctr, y_ctr, width, height, angle])
        rotated_boxes.append(rotated_box)
    return np.array(rotated_boxes)


def poly_to_rotated_box(polys):
    """
    polys:n*8
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    to
    rrect:[x_ctr,y_ctr,w,h,angle]
    """
    pt1, pt2, pt3, pt4 = polys[..., :8].chunk(4, 1)

    edge1 = torch.sqrt(
        torch.pow(pt1[..., 0] - pt2[..., 0], 2) + torch.pow(pt1[..., 1] - pt2[..., 1], 2))
    edge2 = torch.sqrt(
        torch.pow(pt2[..., 0] - pt3[..., 0], 2) + torch.pow(pt2[..., 1] - pt3[..., 1], 2))

    angles1 = torch.atan2((pt2[..., 1] - pt1[..., 1]), (pt2[..., 0] - pt1[..., 0]))
    angles2 = torch.atan2((pt4[..., 1] - pt1[..., 1]), (pt4[..., 0] - pt1[..., 0]))
    angles = polys.new_zeros(polys.shape[0])
    angles[edge1 > edge2] = angles1[edge1 > edge2]
    angles[edge1 <= edge2] = angles2[edge1 <= edge2]

    angles = norm_angle(angles)

    x_ctr = (pt1[..., 0] + pt3[..., 0]) / 2.0
    y_ctr = (pt1[..., 1] + pt3[..., 1]) / 2.0

    edges = torch.stack([edge1, edge2], dim=1)
    width, _ = torch.max(edges, 1)
    height, _ = torch.min(edges, 1)

    return torch.stack([x_ctr, y_ctr, width, height, angles], 1)


def rotated_box_to_bbox_np(rotatex_boxes):
    polys = rotated_box_to_poly_np(rotatex_boxes)
    xmin = polys[:, ::2].min(1, keepdims=True)
    ymin = polys[:, 1::2].min(1, keepdims=True)
    xmax = polys[:, ::2].max(1, keepdims=True)
    ymax = polys[:, 1::2].max(1, keepdims=True)
    return np.concatenate([xmin, ymin, xmax, ymax], axis=1)


def rotated_box_to_bbox(rotatex_boxes):
    polys = rotated_box_to_poly(rotatex_boxes)
    xmin, _ = polys[:, ::2].min(1)
    ymin, _ = polys[:, 1::2].min(1)
    xmax, _ = polys[:, ::2].max(1)
    ymax, _ = polys[:, 1::2].max(1)
    return torch.stack([xmin, ymin, xmax, ymax], dim=1)


def bbox_to_rotated_box(bboxes):
    """
    :param bboxes: shape (n, 4) (xmin, ymin, xmax, ymax) or (n, 5) with score
    :return: dbboxes: shape (n, 5) (x_ctr, y_ctr, w, h, angle)
    """
    num_boxes = bboxes.size(0)

    x_ctr = (bboxes[:, 2] + bboxes[:, 0]) / 2.0
    y_ctr = (bboxes[:, 3] + bboxes[:, 1]) / 2.0
    edges1 = torch.abs(bboxes[:, 2] - bboxes[:, 0])
    edges2 = torch.abs(bboxes[:, 3] - bboxes[:, 1])
    angles = bboxes.new_zeros(num_boxes)

    inds = edges1 < edges2

    if bboxes.size(1) == 4:
        rotated_boxes = torch.stack((x_ctr, y_ctr, edges1, edges2, angles), dim=1)
    # add score dim if exsists
    elif bboxes.size(1) == 5:
        rotated_boxes = torch.stack((x_ctr, y_ctr, edges1, edges2, angles, bboxes[:, 4]), dim=1)
    else:
        return ValueError('bboxes.size(1) must be 4 or 5')

    rotated_boxes[inds, 2] = edges2[inds]
    rotated_boxes[inds, 3] = edges1[inds]
    rotated_boxes[inds, 4] = np.pi / 2.0

    return rotated_boxes


def cal_line_length(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


def get_best_begin_point_single(coordinate):
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
                 [[x3, y3], [x4, y4], [x1, y1], [x2, y2]], [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combinate[i][0], dst_coordinate[0]) \
                     + cal_line_length(combinate[i][1], dst_coordinate[1]) \
                     + cal_line_length(combinate[i][2], dst_coordinate[2]) \
                     + cal_line_length(combinate[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
        # print("choose one direction!")
    return np.array(combinate[force_flag]).reshape(8)


def get_best_begin_point(coordinates):
    coordinates = list(map(get_best_begin_point_single, coordinates.tolist()))
    coordinates = np.array(coordinates)
    return coordinates


def rotated_box_to_roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 5), [batch_ind, x, y, w, h, a]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :5]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 6))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def roi_to_rotated_box(rois):
    bbox_list = []
    img_ids = torch.unique(rois[:, 0].cpu(), sorted=True)
    for img_id in img_ids:
        inds = (rois[:, 0] == img_id.item())
        bbox = rois[inds, 1:]
        bbox_list.append(bbox)
    return bbox_list
