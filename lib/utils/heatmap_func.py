import numpy as np
import torch
import torch.nn.functional as F
import time
import math

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def generate_xy(height, width):
    xi = np.arange(0, width)
    yi = np.arange(0, height)
    xi, yi = np.meshgrid(xi, yi)
    Ys = yi.flatten()
    Xs = xi.flatten()
    xy = np.concatenate([Xs[None, :], Ys[None, :]], axis=0)
    return xy


def gauss2d(xy, amp, x0, y0, a):
    x, y = xy
    inner = a * (np.power(x - x0, 2) + np.power(y-y0, 2))
    return amp * np.exp(-inner)


def exception_loc(heatmap):
    if not isinstance(heatmap, torch.Tensor):
        torch_heatmap = torch.from_numpy(heatmap).type(torch.FloatTensor).cuda()
    else:
        torch_heatmap = heatmap
    normalized_heatmap = torch.pow(torch_heatmap, 2) / torch.sum(torch.pow(torch_heatmap, 2))
    height, width = heatmap.shape
    x, y = generate_xy(height, width)
    y = torch.from_numpy(y).type(torch.FloatTensor).view(height, width).cuda()
    x = torch.from_numpy(x).type(torch.FloatTensor).view(height, width).cuda()
    return torch.sum(normalized_heatmap * x).item(), torch.sum(normalized_heatmap * y).item()


def gaussian_param(heatmap):
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.clone().detach().cpu().numpy()

    heatmap = np.clip(heatmap, 1e-5, 1.)
    init_delta = 8
    init_H, init_W = 64, 48

    height, width = heatmap.shape[-2:]
    delta = 1 / (init_delta * (4 ** math.log((height / init_H), 2)))

    idxes = np.where(heatmap > 1e-3)
    y, x = idxes
    heigh_value_heatmap = heatmap[y, x]

    if not heigh_value_heatmap.any():
        return [0, 0.5, 0.5, 0]
    else:
        loc = exception_loc(heatmap)
        zobs = heigh_value_heatmap.flatten()
        amp = zobs.max()
        return [amp, loc[0], loc[1], delta]


def gaussian_interpolate(heatmap, upscale=1):
    np_heatmap = heatmap.clone().detach().cpu().numpy()
    params = gaussian_param(np_heatmap)
    new_params = [params[0], upscale * params[1], upscale * params[2], params[3] / (2 ** upscale)]
    height, width = np_heatmap.shape[-2:]
    up_height, up_width = height * upscale, width * upscale
    xy = generate_xy(up_height, up_width)
    pred = torch.from_numpy(gauss2d(xy, *new_params)).type(dtype).view(up_height, up_width)
    return pred


def gaussian_sample(heatmap, point_coord):
    np_heatmap = heatmap.clone().detach().cpu().numpy()
    params = gaussian_param(np_heatmap)
    xy = point_coord.clone().permute(1, 0).detach().cpu().numpy()
    height, width = heatmap.shape[-2:]
    xy = xy * torch.tensor([[height], [width]]).type(dtype)
    pred = torch.from_numpy(gauss2d(xy, *params)).type(dtype)
    return pred


def calculate_uncertain_gaussian_heatmap_func(heatamp, upscale=1):
    if len(heatamp.size()) == 3:
        heatamp = torch.squeeze(heatamp, dim=0)

    if upscale == 1:
        gaussian_heatmap = gaussian_interpolate(heatamp, 1)
        diff_map = torch.abs(heatamp - gaussian_heatmap)
    else:
        gaussian_heatmap = gaussian_interpolate(heatamp, upscale)
        interpolated_heatmap = F.interpolate(
            heatamp, scale_factor=upscale, mode="bilinear", align_corners=False
        ).squeeze(0)
        diff_map = torch.abs(interpolated_heatmap - gaussian_heatmap)
    return diff_map
