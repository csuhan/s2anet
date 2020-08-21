import math
import pdb

import numpy as np
import torch

import DOTA_devkit.polyiou as polyiou

from .nms_rotated_C import nms_rotated as nms_rotated_kernel
from mmdet.core import rbox2poly_torch


def pesudo_nms_poly(dets, iou_thr):
    keep = torch.range(0, len(dets))
    return dets, keep


def pesudo_nms_poly_filter_border(dets, iou_thr, max_shape):
    """only consider the tensor case now"""

    xmin, xmin_indices = torch.min(dets[:, :8][:, ::2], 1)
    ymin, ymin_indices = torch.min(dets[:, :8][:, 1::2], 1)
    xmax, xmax_indices = torch.max(dets[:, :8][:, ::2], 1)
    ymax, ymax_indices = torch.max(dets[:, :8][:, 1::2], 1)
    # import pdb;pdb.set_trace()
    keep_inds = (xmin > 10) & (
        xmax < (max_shape[1] - 10)) & (ymin > 10) & (ymax < (max_shape[0] - 10))

    return dets[keep_inds, :], keep_inds


def py_cpu_nms_poly_fast3(dets, iou_thr):
    # TODO: check the type numpy()
    if dets.shape[0] == 0:
        keep = dets.new_zeros(0, dtype=torch.long)
        keep = keep.cpu().numpy()
        device = dets.device
        if isinstance(dets, torch.Tensor):
            dets = dets.cpu().numpy().astype(np.float64)
        if isinstance(iou_thr, torch.Tensor):
            iou_thr = iou_thr.cpu().numpy().astype(np.float64)
    else:
        device = dets.device
        if isinstance(dets, torch.Tensor):
            dets = dets.cpu().numpy().astype(np.float64)
        if isinstance(iou_thr, torch.Tensor):
            iou_thr = iou_thr.cpu().numpy().astype(np.float64)
        obbs = dets[:, 0:-1]
        # pdb.set_trace()
        x1 = np.min(obbs[:, 0::2], axis=1)
        y1 = np.min(obbs[:, 1::2], axis=1)
        x2 = np.max(obbs[:, 0::2], axis=1)
        y2 = np.max(obbs[:, 1::2], axis=1)
        scores = dets[:, 8]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        polys = []
        for i in range(len(dets)):
            tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                               dets[i][2], dets[i][3],
                                               dets[i][4], dets[i][5],
                                               dets[i][6], dets[i][7]])
            polys.append(tm_polygon)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            ovr = []
            i = order[0]
            keep.append(i)
            # if order.size == 0:
            #     break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            # w = np.maximum(0.0, xx2 - xx1 + 1)
            # h = np.maximum(0.0, yy2 - yy1 + 1)
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            hbb_inter = w * h
            hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
            # h_keep_inds = np.where(hbb_ovr == 0)[0]
            h_inds = np.where(hbb_ovr > 0)[0]
            tmp_order = order[h_inds + 1]
            for j in range(tmp_order.size):
                iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
                hbb_ovr[h_inds[j]] = iou

            try:
                if math.isnan(ovr[0]):
                    pdb.set_trace()
            except:
                pass
            inds = np.where(hbb_ovr <= iou_thr)[0]
            order = order[inds + 1]
    return torch.from_numpy(dets[keep, :]).to(device), torch.from_numpy(np.array(keep)).to(device)


def py_cpu_nms_poly_fast3_np(dets, thresh):
    obbs = dets[:, 0:-1]
    x1 = np.min(obbs[:, 0::2], axis=1)
    y1 = np.min(obbs[:, 1::2], axis=1)
    x2 = np.max(obbs[:, 0::2], axis=1)
    y2 = np.max(obbs[:, 1::2], axis=1)
    scores = dets[:, 8]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    polys = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                           dets[i][2], dets[i][3],
                                           dets[i][4], dets[i][5],
                                           dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        # if order.size == 0:
        #     break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # w = np.maximum(0.0, xx2 - xx1 + 1)
        # h = np.maximum(0.0, yy2 - yy1 + 1)
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
        # h_keep_inds = np.where(hbb_ovr == 0)[0]
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou
            # ovr.append(iou)
            # ovr_index.append(tmp_order[j])

        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(hbb_ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def soft_py_cpu_nms_poly_np(
        dets_in,
        iou_thr,
        method=1,
        sigma=0.5,
        min_score=0.05
):

    dets = dets_in.copy()
    N = dets.shape[0]
    inds = np.arange(N)

    for i in range(N):
        maxscore = dets[i, 8]
        maxpos = i

        tx1, ty1, tx2, ty2, tx3, ty3, tx4, ty4 = \
            dets[i, 0], dets[i, 1], dets[i, 2], dets[i,
                                                     3], dets[i, 4], dets[i, 5], dets[i, 6], dets[i, 7]
        ts = dets[i, 8]
        ti = inds[i]

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < dets[pos, 8]:
                maxscore = dets[pos, 8]
                maxpos = pos
            pos = pos + 1

        # add max poly as a detection
        dets[i, 0] = dets[maxpos, 0]
        dets[i, 1] = dets[maxpos, 1]
        dets[i, 2] = dets[maxpos, 2]
        dets[i, 3] = dets[maxpos, 3]
        dets[i, 4] = dets[maxpos, 4]
        dets[i, 5] = dets[maxpos, 5]
        dets[i, 6] = dets[maxpos, 6]
        dets[i, 7] = dets[maxpos, 7]
        dets[i, 8] = dets[maxpos, 8]
        inds[i] = inds[maxpos]

        # swap ith poly with position of max poly
        dets[maxpos, 0] = tx1
        dets[maxpos, 1] = ty1
        dets[maxpos, 2] = tx2
        dets[maxpos, 3] = ty2
        dets[maxpos, 4] = tx3
        dets[maxpos, 5] = ty3
        dets[maxpos, 6] = tx4
        dets[maxpos, 7] = ty4
        dets[maxpos, 8] = ts
        inds[maxpos] = ti

        tx1 = dets[i, 0]
        ty1 = dets[i, 1]
        tx2 = dets[i, 2]
        ty2 = dets[i, 3]
        tx3 = dets[i, 4]
        ty3 = dets[i, 5]
        tx4 = dets[i, 6]
        ty4 = dets[i, 7]
        ts = dets[i, 8]

        # hbb
        txmin, tymin, txmax, tymax = min(tx1, tx2, tx3, tx4), \
            min(ty1, ty2, ty3, ty4), \
            max(tx1, tx2, tx3, tx4), \
            max(ty1, ty2, ty3, ty4)

        pos = i + 1
        # NMS iterations, note that N changes if detection polys fall below threshold
        while pos < N:
            x1, y1, x2, y2, x3, y3, x4, y4 = \
                dets[pos, 0], dets[pos, 1], dets[pos, 2], dets[pos, 3], \
                dets[pos, 4], dets[pos, 5], dets[pos, 6], dets[pos, 7]
            s = dets[pos, 8]

            xmin, ymin, xmax, ymax = min(x1, x2, x3, x4), \
                min(y1, y2, y3, y4), \
                max(x1, x2, x3, x4), \
                max(y1, y2, y3, y4)
            iw = (min(txmax, xmax) - max(txmin, xmin) + 1)
            if iw > 0:
                ih = (min(tymax, ymax) - max(tymin, ymin) + 1)
                if ih > 0:
                    max_polygon = polyiou.VectorDouble(
                        [tx1, ty1, tx2, ty2, tx3, ty3, tx4, ty4])
                    pos_polygon = polyiou.VectorDouble(
                        [x1, y1, x2, y2, x3, y3, x4, y4])
                    ov = polyiou.iou_poly(max_polygon, pos_polygon)
                    if ov > 0:
                        if method == 1:  # linear
                            if ov > iou_thr:
                                weight = 1 - ov
                            else:
                                weight = 1
                        elif method == 2:  # gaussian
                            weight = np.exp(-(ov * ov) / sigma)
                        else:  # original NMS
                            if ov > iou_thr:
                                weight = 0
                            else:
                                weight = 1

                        dets[pos, 8] = weight * dets[pos, 8]

                        # if det score falls below threshold, discard the poly by
                        # swapping with last poly update N
                        if dets[pos, 8] < min_score:
                            dets[pos, 0] = dets[N-1, 0]
                            dets[pos, 1] = dets[N-1, 1]
                            dets[pos, 2] = dets[N-1, 2]
                            dets[pos, 3] = dets[N-1, 3]
                            dets[pos, 4] = dets[N-1, 4]
                            dets[pos, 5] = dets[N-1, 5]
                            dets[pos, 6] = dets[N-1, 6]
                            dets[pos, 7] = dets[N-1, 7]
                            dets[pos, 8] = dets[N-1, 8]
                            inds[pos] = inds[N - 1]
                            N = N - 1
                            pos = pos - 1
            pos = pos + 1

    return inds[:N]


def soft_py_cpu_nms_poly(
        dets,
        iou_thr,
        method=1,
        sigma=0.5,
        min_score=0.001
):
    # TODO: 1. check the correctness
    # TODO: 2. optimize it
    if dets.shape[0] == 0:
        keep = dets.new_zeros(0, dtype=torch.long)
        keep = keep.cpu().numpy()
        device = dets.device
        if isinstance(dets, torch.Tensor):
            dets.dets.cpu().numpy().astype(np.float64)
        if isinstance(iou_thr, torch.Tensor):
            iou_thr = iou_thr.cpu().numpy().astype(np.float64)
    else:
        device = dets.device
        if isinstance(dets, torch.Tensor):
            dets = dets.cpu().numpy().astype(np.float64)
        if isinstance(iou_thr, torch.Tensor):
            iou_thr = iou_thr.cpu().numpy().astype(np.float64)
        N = dets.shape[0]
        inds = np.arange(N)

        for i in range(N):
            maxscore = dets[i, 8]
            maxpos = i

            tx1, ty1, tx2, ty2, tx3, ty3, tx4, ty4 = \
                dets[i, 0], dets[i, 1], dets[i, 2], dets[i,
                                                         3], dets[i, 4], dets[i, 5], dets[i, 6], dets[i, 7]
            ts = dets[i, 8]
            ti = inds[i]

            pos = i + 1
            # get max box
            while pos < N:
                if maxscore < dets[pos, 8]:
                    maxscore = dets[pos, 8]
                    maxpos = pos
                pos = pos + 1

            # add max poly as a detection
            dets[i, 0] = dets[maxpos, 0]
            dets[i, 1] = dets[maxpos, 1]
            dets[i, 2] = dets[maxpos, 2]
            dets[i, 3] = dets[maxpos, 3]
            dets[i, 4] = dets[maxpos, 4]
            dets[i, 5] = dets[maxpos, 5]
            dets[i, 6] = dets[maxpos, 6]
            dets[i, 7] = dets[maxpos, 7]
            dets[i, 8] = dets[maxpos, 8]
            inds[i] = inds[maxpos]

            # swap ith poly with position of max poly
            dets[maxpos, 0] = tx1
            dets[maxpos, 1] = ty1
            dets[maxpos, 2] = tx2
            dets[maxpos, 3] = ty2
            dets[maxpos, 4] = tx3
            dets[maxpos, 5] = ty3
            dets[maxpos, 6] = tx4
            dets[maxpos, 7] = ty4
            dets[maxpos, 8] = ts
            inds[maxpos] = ti

            tx1 = dets[i, 0]
            ty1 = dets[i, 1]
            tx2 = dets[i, 2]
            ty2 = dets[i, 3]
            tx3 = dets[i, 4]
            ty3 = dets[i, 5]
            tx4 = dets[i, 6]
            ty4 = dets[i, 7]
            ts = dets[i, 8]

            pos = i + 1
            # NMS iterations, note that N changes if detection polys fall below threshold
            while pos < N:
                x1, y1, x2, y2, x3, y3, x4, y4 = \
                    dets[pos, 0], dets[pos, 1], dets[pos, 2], dets[pos, 3], \
                    dets[pos, 4], dets[pos, 5], dets[pos, 6], dets[pos, 7]
                s = dets[pos, -1]

                # TODO:finish the area calculation
                # area = ()
                # finish the iou calculation
                max_polygon = polyiou.VectorDouble(
                    [tx1, ty1, tx2, ty2, tx3, ty3, tx4, ty4])
                pos_polygon = polyiou.VectorDouble(
                    [x1, y1, x2, y2, x3, y3, x4, y4])
                ov = polyiou.iou_poly(max_polygon, pos_polygon)

                if method == 1:  # linear
                    if ov > iou_thr:
                        weight = 1 - ov
                    else:
                        weight = 1
                elif method == 2:  # gaussian
                    weight = np.exp(-(ov * ov) / sigma)
                else:  # original NMS
                    if ov > iou_thr:
                        weight = 0
                    else:
                        weight = 1

                dets[pos, 8] = weight * dets[pos, 8]

                # if det score falls below threshold, discard the poly by
                # swapping with last poly update N
                if dets[pos, 8] < min_score:
                    dets[pos, 0] = dets[N-1, 0]
                    dets[pos, 1] = dets[N-1, 1]
                    dets[pos, 2] = dets[N-1, 2]
                    dets[pos, 3] = dets[N-1, 3]
                    dets[pos, 4] = dets[N-1, 4]
                    dets[pos, 5] = dets[N-1, 5]
                    dets[pos, 6] = dets[N-1, 6]
                    dets[pos, 7] = dets[N-1, 7]
                    dets[pos, 8] = dets[N-1, 8]
                    inds[pos] = inds[N - 1]
                    N = N - 1
                    pos = pos - 1
                pos = pos + 1

    return torch.from_numpy(dets[:N]).to(device), torch.from_numpy(inds[:N]).to(device)


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    #print('dets:', dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # index for dets
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def nms_rotated(dets, iou_thr):
    if dets.shape[0] == 0:
        return dets
    keep_inds = nms_rotated_kernel(dets[:, :5], dets[:, 5], iou_thr)
    dets = dets[keep_inds, :]
    return dets, keep_inds

def cpu_soft_nms_rbox(
        dets,
        iou_thr,
        method=1,
        sigma=0.5,
        min_score=0.001):
    rboxes = dets[:,:5]
    scores = dets[:,5]
    polys = rbox2poly_torch(rboxes)
    dets = torch.cat([polys,scores[:,None]],dim=1)
    return soft_py_cpu_nms_poly(dets,iou_thr,method,sigma,min_score)