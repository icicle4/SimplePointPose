import numpy as np
import torch
import torch.nn.functional as F
import scipy.optimize as opt

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def generate_xy(height, width):

    xi = (torch.linspace(0, 1, width + 1)[:-1].type(dtype) + .5) / width
    yi = (torch.linspace(0, 1, height + 1)[:-1].type(dtype) + .5) / height
    xi, yi = torch.meshgrid(xi, yi)

    Ys = yi.permute(1, 0).flatten()
    Xs = xi.permute(1, 0).flatten()
    xy = torch.cat([Xs[None, :], Ys[None, :]], dim=0)
    return xy


def gaussian_param(heatmap):
    heatmap = torch.clamp(heatmap, 1e-5, 1.)
    height, width = heatmap.shape
    xy = generate_xy(height, width)
    zobs = heatmap.view(-1)
    i = zobs.argmax()
    x, y = xy

    x0, y0 = x[i], y[i]
    M = generate_M(xy, x0, y0, height, width)
    amp = zobs.max()
    target = -torch.log(zobs / amp)
    torch_res = torch.lstsq(target.view(-1, 1), M)[0]
    a, c = torch_res[:2]
    return [amp, x0, y0, a, c]


def gauss2d(xy, amp, x0, y0, a, c):
    x, y = xy
    inner = a * torch.pow(x - x0, 2)
    inner += 2 * c * (x - x0) * (y - y0)
    inner += a * torch.pow(y-y0, 2)
    return amp * torch.exp(-inner)


def gaussian_interpolate(heatmap, upscale=1):
    try:
        height, width = heatmap.shape[-2:]
        up_height, up_width = height * upscale, width * upscale
        params = gaussian_param(heatmap)
        xy = generate_xy(up_height, up_width)
        zpred = gauss2d(xy, *params).view(up_height, up_width)
    except RuntimeError as e:
        if upscale == 1:
            zpred = heatmap
        else:
            zpred = F.interpolate(
                heatmap.unsqueeze(0), scale_factor=upscale, mode="bilinear", align_corners=False
            ).squeeze(0)
    return zpred


def gaussian_sample(heatmap, point_coord):
    try:
        params = gaussian_param(heatmap)
        xy = point_coord.clone().permute(1, 0)
        zpred = gauss2d(xy, *params)
    except RuntimeError as e:
        zpred = F.grid_sample(heatmap[None, None, :, :], 2 * point_coord[None, None, :, :] - 1.0).squeeze()
    return zpred


def generate_M(xy, x0, y0, height, width):
    x, y = xy
    M = torch.zeros((height * width, 2)).type(dtype)
    M[:, 0] = torch.pow(x - x0, 2) + torch.pow(y - y0, 2)
    M[:, 1] = 2 * (x-x0) * (y-y0)
    return M

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
