
import math

import mmcv
import numpy as np
import torch

PI = np.pi


def rbox2delta(proposals, gt, means=[0, 0, 0, 0, 0], stds=[1, 1, 1, 1, 1]):
    gt_widths = gt[..., 2]
    gt_heights = gt[..., 3]
    gt_angle = gt[..., 4]

    proposals_widths = proposals[..., 2]
    proposals_heights = proposals[..., 3]
    proposals_angle = proposals[..., 4]

    coord = gt[..., 0:2] - proposals[..., 0:2]
    dx = (torch.cos(proposals[..., 4]) * coord[..., 0] +
          torch.sin(proposals[..., 4]) * coord[..., 1]) / proposals_widths
    dy = (-torch.sin(proposals[..., 4]) * coord[..., 0] +
          torch.cos(proposals[..., 4]) * coord[..., 1]) / proposals_heights
    dw = torch.log(gt_widths / proposals_widths)
    dh = torch.log(gt_heights / proposals_heights)
    da = (gt_angle - proposals_angle)

    da = (da + PI / 4) % PI - PI / 4
    da /= PI

    deltas = torch.stack((dx, dy, dw, dh, da), -1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2rbox(Rrois,
               deltas,
               means=[0, 0, 0, 0, 0],
               stds=[1, 1, 1, 1, 1],
               max_shape=None,
               wh_ratio_clip=16 / 1000):
    """

    :param Rrois: (cx, cy, w, h, theta)
    :param deltas: (dx, dy, dw, dh, dtheta)
    :param means:
    :param stds:
    :param max_shape:
    :param wh_ratio_clip:
    :return:
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
    Rroi_x = (Rrois[:, 0]).unsqueeze(1).expand_as(dx)
    Rroi_y = (Rrois[:, 1]).unsqueeze(1).expand_as(dy)
    Rroi_w = (Rrois[:, 2]).unsqueeze(1).expand_as(dw)
    Rroi_h = (Rrois[:, 3]).unsqueeze(1).expand_as(dh)
    Rroi_angle = (Rrois[:, 4]).unsqueeze(1).expand_as(dangle)
    gx = dx * Rroi_w * torch.cos(Rroi_angle) \
        - dy * Rroi_h * torch.sin(Rroi_angle) + Rroi_x
    gy = dx * Rroi_w * torch.sin(Rroi_angle) \
        + dy * Rroi_h * torch.cos(Rroi_angle) + Rroi_y
    gw = Rroi_w * dw.exp()
    gh = Rroi_h * dh.exp()

    ga = np.pi * dangle + Rroi_angle
    ga = (ga + PI / 4) % PI - PI / 4

    bboxes = torch.stack([gx, gy, gw, gh, ga], dim=-1).view_as(deltas)
    return bboxes


def rbox_flip(bboxes, img_shape):
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
        angle = (angle + PI / 4) % PI - PI / 4
        flipped[:, 4::5] = angle
        return flipped
    elif isinstance(bboxes, np.ndarray):
        flipped = bboxes.copy()
        # flip x
        flipped[..., 0::5] = img_shape[1] - bboxes[..., 0::5] - 1
        # flip angle
        angle = -bboxes[..., 4::5]
        angle = (angle + PI / 4) % PI - PI / 4
        flipped[..., 4::5] = angle
        return flipped


def rbox_mapping(dbboxes, img_shape, scale_factor, flip):
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
        new_dbboxes = rbox_flip(new_dbboxes, img_shape)

    return new_dbboxes

# Test passed


def rbox_mapping_back(dbboxes, img_shape, scale_factor, flip):
    """
    Map dbboxes from testing scael to original image scale
    :param dbboxes:
    :param img_shape:
    :param scale_factor:
    :param flip:
    :return:
    """
    new_dbboxes = rbox_flip(dbboxes, img_shape) if flip else dbboxes
    new_dbboxes[..., 0::5] = new_dbboxes[..., 0::5] / scale_factor
    new_dbboxes[..., 1::5] = new_dbboxes[..., 1::5] / scale_factor
    new_dbboxes[..., 2::5] = new_dbboxes[..., 2::5] / scale_factor
    new_dbboxes[..., 3::5] = new_dbboxes[..., 3::5] / scale_factor
    return new_dbboxes


def rbox2result(bboxes, labels, num_classes):
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


def rbox2poly_single(rrect):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    x_ctr, y_ctr, width, height, angle = rrect[:5]
    tl_x, tl_y, br_x, br_y = -width/2, -height/2, width/2, height/2
    rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    poly = R.dot(rect)
    x0, x1, x2, x3 = poly[0, :4] + x_ctr
    y0, y1, y2, y3 = poly[1, :4] + y_ctr
    poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3], dtype=np.float32)
    poly = get_best_begin_point_single(poly)
    return poly


def rbox2poly(rrects):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    polys = []
    for rrect in rrects:
        x_ctr, y_ctr, width, height, angle = rrect[:5]
        tl_x, tl_y, br_x, br_y = -width/2, -height/2, width/2, height/2
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


def rbox2poly_torch(rboxes):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    N = rboxes.shape[0]
    x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
        1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
    tl_x, tl_y, br_x, br_y = -width*0.5, -height*0.5, width*0.5, height*0.5

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


def poly2rbox_single(poly):
    """
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    to
    rrect:[x_ctr,y_ctr,w,h,angle]
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

    angle = 0
    width = 0
    height = 0

    if edge1 > edge2:
        width = edge1
        height = edge2
        angle = np.arctan2(
            np.float(pt2[1] - pt1[1]), np.float(pt2[0] - pt1[0]))
    elif edge2 >= edge1:
        width = edge2
        height = edge1
        angle = np.arctan2(
            np.float(pt4[1] - pt1[1]), np.float(pt4[0] - pt1[0]))

    angle = (angle + PI / 4) % PI - PI / 4

    x_ctr = np.float(pt1[0] + pt3[0]) / 2
    y_ctr = np.float(pt1[1] + pt3[1]) / 2
    rrect = np.array([x_ctr, y_ctr, width, height, angle])
    return rrect


def poly2rbox(polys):
    """
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    to
    rrect:[x_ctr,y_ctr,w,h,angle]
    """
    rrects = []
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

        angle = 0
        width = 0
        height = 0

        if edge1 > edge2:
            width = edge1
            height = edge2
            angle = np.arctan2(
                np.float(pt2[1] - pt1[1]), np.float(pt2[0] - pt1[0]))
        elif edge2 >= edge1:
            width = edge2
            height = edge1
            angle = np.arctan2(
                np.float(pt4[1] - pt1[1]), np.float(pt4[0] - pt1[0]))

        angle = (angle + PI / 4) % PI - PI / 4

        x_ctr = np.float(pt1[0] + pt3[0]) / 2
        y_ctr = np.float(pt1[1] + pt3[1]) / 2
        rrect = np.array([x_ctr, y_ctr, width, height, angle])
        rrects.append(rrect)
    return np.array(rrects)


def poly2rbox_torch(polys):
    """
    polys:n*8
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    to
    rrect:[x_ctr,y_ctr,w,h,angle]
    """
    pt1, pt2, pt3, pt4 = polys[..., :8].chunk(4, 1)

    edge1 = torch.sqrt(
        torch.pow(pt1[..., 0]-pt2[..., 0], 2)+torch.pow(pt1[..., 1]-pt2[..., 1], 2))
    edge2 = torch.sqrt(
        torch.pow(pt2[..., 0]-pt3[..., 0], 2)+torch.pow(pt2[..., 1]-pt3[..., 1], 2))

    angles1 = torch.atan2((pt2[..., 1]-pt1[..., 1]), (pt2[..., 0]-pt1[..., 0]))
    angles2 = torch.atan2((pt4[..., 1]-pt1[..., 1]), (pt4[..., 0]-pt1[..., 0]))
    angles = polys.new_zeros(polys.shape[0])
    angles[edge1 > edge2] = angles1[edge1 > edge2]
    angles[edge1 <= edge2] = angles2[edge1 <= edge2]

    angles = (angles + PI / 4) % PI - PI / 4

    x_ctr = (pt1[..., 0] + pt3[..., 0]) / 2.0
    y_ctr = (pt1[..., 1] + pt3[..., 1]) / 2.0

    edges = torch.stack([edge1, edge2], dim=1)
    width, _ = torch.max(edges, 1)
    height, _ = torch.min(edges, 1)

    return torch.stack([x_ctr, y_ctr, width, height, angles], 1)


def rbox2rect(rrects):
    polys = rbox2poly(rrects)
    xmin = polys[:, ::2].min(1, keepdims=True)
    ymin = polys[:, 1::2].min(1, keepdims=True)
    xmax = polys[:, ::2].max(1, keepdims=True)
    ymax = polys[:, 1::2].max(1, keepdims=True)
    return np.concatenate([xmin, ymin, xmax, ymax], axis=1)


def rbox2rect_torch(rrects):
    polys = rbox2poly_torch(rrects)
    xmin, _ = polys[:, ::2].min(1)
    ymin, _ = polys[:, 1::2].min(1)
    xmax, _ = polys[:, ::2].max(1)
    ymax, _ = polys[:, 1::2].max(1)
    return torch.stack([xmin, ymin, xmax, ymax], dim=1)


def rect2rbox(bboxes):
    """
    :param bboxes: shape (n, 4) (xmin, ymin, xmax, ymax)
    :return: dbboxes: shape (n, 5) (x_ctr, y_ctr, w, h, angle)
    """
    num_boxes = bboxes.size(0)

    x_ctr = (bboxes[:, 2]+bboxes[:, 0]) / 2.0
    y_ctr = (bboxes[:, 3]+bboxes[:, 1]) / 2.0
    edges1 = torch.abs(bboxes[:, 2]-bboxes[:, 0])
    edges2 = torch.abs(bboxes[:, 3]-bboxes[:, 1])
    angles = bboxes.new_zeros(num_boxes)

    inds = edges1 < edges2

    rboxes = torch.stack((x_ctr, y_ctr, edges1, edges2, angles), dim=1)
    rboxes[inds, 2] = edges2[inds]
    rboxes[inds, 3] = edges1[inds]
    rboxes[inds, 4] = np.pi / 2.0

    return rboxes


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


def get_best_begin_point_torch(coordinates):
    """
    coordinates: 8 points
    """
    device = coordinates.device
    dtype = coordinates.dtype
    coordinates = coordinates.cpu().numpy()
    coordinates = get_best_begin_point(coordinates)
    coordinates = torch.tensor(coordinates, dtype=dtype, device=device)
    return coordinates


def cal_line_length_torch(point1, point2):
    """
    point1:[x,y]
    """
    return torch.sqrt(torch.pow(point1[0] - point2[0], 2) + torch.pow(point1[1] - point2[1], 2))


def rbox2roi(bbox_list):
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


def roi2rbox(rois):
    bbox_list = []
    img_ids = torch.unique(rois[:, 0].cpu(), sorted=True)
    for img_id in img_ids:
        inds = (rois[:, 0] == img_id.item())
        bbox = rois[inds, 1:]
        bbox_list.append(bbox)
    return bbox_list
