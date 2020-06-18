import torch
from torch.nn import functional as F
from utils import heatmap_func
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes


def generate_regular_grid_point_coords(R, side_size, device):
    """
    Generate regular square grid of points in [0, 1] x [0, 1] coordinate space.
    Args:
        R (int): The number of grids to sample, one for each region.
        side_size (int): The side size of the regular grid.
        device (torch.device): Desired device of returned tensor.
    Returns:
        (Tensor): A tensor of shape (R, side_size^2, 2) that contains coordinates
            for the regular grids.
    """
    aff = torch.tensor([[[0.5, 0, 0.5], [0, 0.5, 0.5]]], device=device)
    r = F.affine_grid(aff, torch.Size((1, 1, side_size, side_size)), align_corners=False)
    return r.view(1, -1, 2).expand(R, -1, -1)


def point_sample(input, point_coords, **kwargs):
    add_dim = False

    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)

    output = F.grid_sample(input, 2 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def get_interest_box(heatmap, k=49):
    height, width = heatmap.size()
    values, idxs = torch.topk(heatmap.flatten(), k=k)

    min_idx, max_idx = min(idxs), max(idxs)
    min_x, min_y = min_idx % width, min_idx // width
    max_x, max_y = max_idx % width, max_idx // width
    bbox = torch.tensor([min_x, min_y, max_x, max_y]).long().cuda().unsqueeze(dim=0)


    return bbox


def get_certain_point_coors_with_randomness(
    coarse_heatmaps, certainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_sampled = int(num_points * oversample_ratio)
    D, C, H, W = coarse_heatmaps.size()
    flatten_coarse_heatmaps = coarse_heatmaps.view(D * C, 1, H, W)
    target_num = flatten_coarse_heatmaps.shape[0]
    proposal_boxes = [get_interest_box(heatmap[0]) for heatmap in flatten_coarse_heatmaps]

    point_coords = torch.rand(target_num, num_sampled, 2, device=coarse_heatmaps.device)
    #print('point coords', point_coords)

    cat_boxes = torch.cat(proposal_boxes)

    p_coords = point_coords.clone()
    point_coords_wrt_heatmap = get_point_coords_wrt_roi(cat_boxes, p_coords)

    point_logits = []

    for i in range(target_num):
        coarse_heatmap = flatten_coarse_heatmaps[i]
        h, w = coarse_heatmap.shape[-2:]
        uncertain_map = heatmap_func.calculate_uncertain_gaussian_heatmap_func(coarse_heatmap, upscale=1).unsqueeze(0)
        point_coords_scaled = point_coords_wrt_heatmap[i] / torch.tensor([w, h], device=coarse_heatmap.device)

        point_logits.append(
            point_sample(
                uncertain_map.unsqueeze(0),
                point_coords_scaled.unsqueeze(0),
                align_corners=False
            ).squeeze(0)
        )

    point_logits = torch.cat(point_logits, dim=0).unsqueeze(dim=1)
    #print('point_logits', point_logits.size(), point_logits.type())
    point_uncertainties = certainty_func(point_logits)

    #print('point_uncertainties', point_uncertainties.size(), point_uncertainties.type())

    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(target_num, dtype=torch.long, device=coarse_heatmaps.device)
    idx += shift[:, None]

    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        target_num, num_uncertain_points, 2
    )
    #print('point_coords', point_coords.size(), point_coords.type(), point_coords.device)

    if num_random_points > 0:
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(target_num, num_random_points, 2, device=coarse_heatmaps.device),
            ],
            dim=1,
        )

    return proposal_boxes, point_coords


def calculate_certainty(logits, classes):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, C, ...) or (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
        classes (list): A list of length R that contains either predicted of ground truth class
            for eash predicted mask.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    if logits.shape[1] == 1:
        gt_class_logits = logits.clone().cuda()
    else:
        gt_class_logits = logits[
            torch.arange(logits.shape[0], device=logits.device), classes
        ].unsqueeze(1).cuda()

    return torch.abs(gt_class_logits)


def point_sample_pd_heatmaps(pd_heatmaps, point_coords_wrt_heatmap, scale = 1):
    """
    :param pd_heatmaps: D, C, H, W
    :param point_coords_wrt_heatmap: C, D, num_sampled, 2
    :return:
    """
    D, C, H, W = pd_heatmaps.size()
    point_logits = []

    point_coords_scaled = point_coords_wrt_heatmap / (torch.tensor([W, H], device=pd_heatmaps.device) * scale)

    for i in range(C):
        point_coord_scaled = point_coords_scaled[i]
        pd_heatmap = pd_heatmaps[:, i:i+1]
        point_logits.append(
            point_sample(
                pd_heatmap,
                point_coord_scaled,
                align_corners=False
            )
        )
    point_logits = torch.cat(point_logits, dim=0)
    return point_logits


def point_sample_pd_features(pd_heatmaps, point_coords):
    point_logits = []
    D, C, _, _ = pd_heatmaps.shape
    for i in range(C):
        point_coord = point_coords[i]
        pd_heatmap = pd_heatmaps[:, i:i+1]
        point_logits.append(
            point_sample(
                pd_heatmap,
                point_coord,
                align_corners=False
            )
        )
    point_logits = torch.cat(point_logits, dim=0)
    return point_logits


def point_sample_features(feature, point_coords):
    point_logits = []

    C, D, num_sampled, _ = point_coords.shape

    for i in range(C):
        point_coord = point_coords[i]
        sample_logit = point_sample(
            feature,
            point_coord,
            align_corners=False
        )
        point_logits.append(
            sample_logit
        )
    point_logits = torch.cat(point_logits, dim=0)
    return point_logits



def point_sample_fine_grained_features(feature, point_coords_wrt_heatmap, scale=1):
    point_logits = []

    H, W = feature.shape[-2:]
    point_coords_scaled = point_coords_wrt_heatmap / (torch.tensor([W, H], device=feature.device) * scale)
    C, D, num_sampled, _ = point_coords_wrt_heatmap.shape

    for i in range(C):
        point_coord_scaled = point_coords_scaled[i]
        sample_logit = point_sample(
                feature,
                point_coord_scaled,
                align_corners=False
            )
        point_logits.append(
            sample_logit
        )

    point_logits = torch.cat(point_logits, dim=0)
    return point_logits


def get_point_coords_wrt_roi(boxes_coords, point_coords):
    """
    Convert box-normalized [0, 1] x [0, 1] point cooordinates to image-level coordinates.
    Args:
        boxes_coords (Tensor): A tensor of shape (R, 4) that contains bounding boxes.
            coordinates.
        point_coords (Tensor): A tensor of shape (R, P, 2) that contains
            [0, 1] x [0, 1] box-normalized coordinates of the P sampled points.
    Returns:
        point_coords_wrt_image (Tensor): A tensor of shape (R, P, 2) that contains
            image-normalized coordinates of P sampled points.
    """
    with torch.no_grad():
        point_coords_wrt_image = point_coords.clone()
        #print('point coords wrt', point_coords_wrt_image)
        point_coords_wrt_image[:, :, 0] = point_coords_wrt_image[:, :, 0] * (
            boxes_coords[:, None, 2] - boxes_coords[:, None, 0]
        )
        point_coords_wrt_image[:, :, 1] = point_coords_wrt_image[:, :, 1] * (
            boxes_coords[:, None, 3] - boxes_coords[:, None, 1]
        )
        point_coords_wrt_image[:, :, 0] += boxes_coords[:, None, 0]
        point_coords_wrt_image[:, :, 1] += boxes_coords[:, None, 1]
    return point_coords_wrt_image



def get_certain_point_coords_on_grid(certainty_map, num_points):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.
    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains uncertainty
            values for a set of points on a regular H x W grid.
        num_points (int): The number of points P to select.
    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W grid.
    """
    R, _, H, W = certainty_map.shape
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)

    num_points = min(H * W, num_points)
    point_indices = torch.topk(certainty_map.view(R, H * W), k=num_points, dim=1)[1]
    point_coords = torch.zeros(R, num_points, 2, dtype=torch.float)
    point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
    point_coords[:, :, 1] = h_step / 2.0 + (point_indices // W).to(torch.float) * h_step
    return point_indices.cuda(), point_coords.cuda()