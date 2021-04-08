'''
torch implementation of 2d oriented box intersection
author: lanxiao li
Modified by csuhan: 
    Remove the `batch` indice in a tensor. 
    This setting is more suitable for mmdet.
'''
import torch

from .sort_vertices_cuda import sort_vertices_forward

EPSILON = 1e-8


def get_intersection_points(polys1: torch.Tensor, polys2: torch.Tensor):
    """Find intersection points of rectangles
    Convention: if two edges are collinear, there is no intersection point

    Args:
        polys1 (torch.Tensor): n, 4, 2
        polys2 (torch.Tensor): n, 4, 2

    Returns:
        intersectons (torch.Tensor): n, 4, 4, 2
        mask (torch.Tensor) : n, 4, 4; bool
    """
    # build edges from corners
    line1 = torch.cat([polys1, polys1[..., [1, 2, 3, 0], :]],
                      dim=2)  # n, 4, 4: Box, edge, point
    line2 = torch.cat([polys2, polys2[..., [1, 2, 3, 0], :]], dim=2)
    # duplicate data to pair each edges from the boxes
    # (n, 4, 4) -> (n, 4, 4, 4) : Box, edge1, edge2, point
    line1_ext = line1.unsqueeze(2).repeat([1, 1, 4, 1])
    line2_ext = line2.unsqueeze(1).repeat([1, 4, 1, 1])
    x1 = line1_ext[..., 0]
    y1 = line1_ext[..., 1]
    x2 = line1_ext[..., 2]
    y2 = line1_ext[..., 3]
    x3 = line2_ext[..., 0]
    y3 = line2_ext[..., 1]
    x4 = line2_ext[..., 2]
    y4 = line2_ext[..., 3]
    # math: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    num = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    den_t = (x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)
    t = den_t / num
    t[num == .0] = -1.
    mask_t = (t > 0) * (t < 1)                # intersection on line segment 1
    den_u = (x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)
    u = -den_u / num
    u[num == .0] = -1.
    mask_u = (u > 0) * (u < 1)                # intersection on line segment 2
    mask = mask_t * mask_u
    # overwrite with EPSILON. otherwise numerically unstable
    t = den_t / (num + EPSILON)
    intersections = torch.stack([x1 + t*(x2-x1), y1 + t*(y2-y1)], dim=-1)
    intersections = intersections * mask.float().unsqueeze(-1)
    return intersections, mask


def get_in_box_points(polys1: torch.Tensor, polys2: torch.Tensor):
    """check if corners of poly1 lie in poly2
    Convention: if a corner is exactly on the edge of the other box, it's also a valid point

    Args:
        polys1 (torch.Tensor): (n, 4, 2)
        polys2 (torch.Tensor): (n, 4, 2)

    Returns:
        c1_in_2: (n, 4) Bool
    """
    a = polys2[..., 0:1, :]  # (n, 1, 2)
    b = polys2[..., 1:2, :]  # (n, 1, 2)
    d = polys2[..., 3:4, :]  # (n, 1, 2)
    ab = b - a                  # (n, 1, 2)
    am = polys1 - a           # (n, 4, 2)
    ad = d - a                  # (n, 1, 2)
    p_ab = torch.sum(ab * am, dim=-1)       # (n, 4)
    norm_ab = torch.sum(ab * ab, dim=-1)    # (n, 1)
    p_ad = torch.sum(ad * am, dim=-1)       # (n, 4)
    norm_ad = torch.sum(ad * ad, dim=-1)    # (n, 1)
    # NOTE: the expression looks ugly but is stable if the two boxes are exactly the same
    # also stable with different scale of bboxes
    cond1 = (p_ab / norm_ab > - 1e-6) * \
        (p_ab / norm_ab < 1 + 1e-6)   # (n, 4)
    cond2 = (p_ad / norm_ad > - 1e-6) * \
        (p_ad / norm_ad < 1 + 1e-6)   # (n, 4)
    return cond1 * cond2


def build_vertices(polys1: torch.Tensor, polys2: torch.Tensor,
                   c1_in_2: torch.Tensor, c2_in_1: torch.Tensor,
                   inters: torch.Tensor, mask_inter: torch.Tensor):
    """find vertices of intersection area

    Args:
        polys1 (torch.Tensor): (n, 4, 2)
        polys2 (torch.Tensor): (n, 4, 2)
        c1_in_2 (torch.Tensor): Bool, (n, 4)
        c2_in_1 (torch.Tensor): Bool, (n, 4)
        inters (torch.Tensor): (n, 4, 4, 2)
        mask_inter (torch.Tensor): (n, 4, 4)

    Returns:
        vertices (torch.Tensor): (n, 24, 2) vertices of intersection area. only some elements are valid
        mask (torch.Tensor): (n, 24) indicates valid elements in vertices
    """
    # NOTE: inter has elements equals zero and has zeros gradient (masked by multiplying with 0).
    # can be used as trick
    n = polys1.size(0)
    # (n, 4+4+16, 2)
    vertices = torch.cat([polys1, polys2, inters.view(
        [n, -1, 2])], dim=1)
    # Bool (n, 4+4+16)
    mask = torch.cat([c1_in_2, c2_in_1, mask_inter.view([n, -1])], dim=1)
    return vertices, mask


def sort_indices(vertices: torch.Tensor, mask: torch.Tensor):
    """[summary]

    Args:
        vertices (torch.Tensor): float (n, 24, 2)
        mask (torch.Tensor): bool (n, 24)

    Returns:
        sorted_index: bool (n, 9)

    Note:
        why 9? the polygon has maximal 8 vertices. +1 to duplicate the first element.
        the index should have following structure:
            (A, B, C, ... , A, X, X, X) 
        and X indicates the index of arbitary elements in the last 16 (intersections not corners) with 
        value 0 and mask False. (cause they have zero value and zero gradient)
    """
    # here we pad dim 0 to be consistent with the `sort_vertices_forward` function
    vertices = vertices.unsqueeze(0)
    mask = mask.unsqueeze(0)

    num_valid = torch.sum(mask.int(), dim=2).int()      # (B, N)
    mean = torch.sum(vertices * mask.float().unsqueeze(-1), dim=2,
                     keepdim=True) / num_valid.unsqueeze(-1).unsqueeze(-1)
    # normalization makes sorting easier
    vertices_normalized = vertices - mean
    return sort_vertices_forward(vertices_normalized, mask, num_valid).squeeze(0).long()


def calculate_area(idx_sorted: torch.Tensor, vertices: torch.Tensor):
    """calculate area of intersection

    Args:
        idx_sorted (torch.Tensor): (n, 9)
        vertices (torch.Tensor): (n, 24, 2)

    return:
        area: (n), area of intersection
        selected: (n, 9, 2), vertices of polygon with zero padding 
    """
    idx_ext = idx_sorted.unsqueeze(-1).repeat([1, 1, 2])
    selected = torch.gather(vertices, 1, idx_ext)
    total = selected[..., 0:-1, 0]*selected[..., 1:, 1] - \
        selected[..., 0:-1, 1]*selected[..., 1:, 0]
    total = torch.sum(total, dim=1)
    area = torch.abs(total) / 2
    return area, selected


def oriented_box_intersection_2d(polys1: torch.Tensor, polys2: torch.Tensor):
    """calculate intersection area of 2d rectangles 

    Args:
        polys1 (torch.Tensor): (n, 4, 2)
        polys2 (torch.Tensor): (n, 4, 2)

    Returns:
        area: (n,), area of intersection
        selected: (n, 9, 2), vertices of polygon with zero padding 
    """
    # find intersection points
    inters, mask_inter = get_intersection_points(polys1, polys2)
    # find inter points
    c12 = get_in_box_points(polys1, polys2)
    c21 = get_in_box_points(polys2, polys1)
    # build vertices
    vertices, mask = build_vertices(
        polys1, polys2, c12, c21, inters, mask_inter)
    # getting sorted indices
    sorted_indices = sort_indices(vertices, mask)
    # calculate areas using torch.gather
    return calculate_area(sorted_indices, vertices)
