import numpy as np
import torch
import torch.nn.functional as F
import time

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def generate_xy(height, width):
    xi = (np.linspace(0, 1, width + 1)[:-1] + .5) / width
    yi = (np.linspace(0, 1, height + 1)[:-1] + .5) / height
    xi, yi = np.meshgrid(xi, yi)
    Ys = yi.flatten()
    Xs = xi.flatten()
    xy = np.concatenate([Xs[None, :], Ys[None, :]], axis=0)
    return xy


def gaussian_param(heatmap):
    heatmap = np.clip(heatmap, 1e-5, 1.)
    height, width = heatmap.shape
    xy = generate_xy(height, width)
    zobs = heatmap.flatten()
    i = zobs.argmax()
    x, y = xy

    x0, y0 = x[i], y[i]
    M = generate_M(xy, x0, y0, height, width)
    amp = zobs.max()
    target = -np.log(zobs / amp)
    a, c = np.linalg.lstsq(M, target, rcond=None)[0]
    return [amp, x0, y0, a, c]


def gauss2d(xy, amp, x0, y0, a, c):
    x, y = xy
    inner = a * np.power(x - x0, 2)
    inner += 2 * c * (x - x0) * (y - y0)
    inner += a * np.power(y-y0, 2)
    return amp * np.exp(-inner)


def gaussian_interpolate(heatmap, upscale=1):
    try:
        time1 = time.time()
        np_heatmap = heatmap.clone().detach().cpu().numpy()
        height, width = np_heatmap.shape[-2:]
        up_height, up_width = height * upscale, width * upscale
        params = gaussian_param(np_heatmap)
        xy = generate_xy(up_height, up_width)
        zpred = torch.from_numpy(gauss2d(xy, *params)).type(dtype).view(up_height, up_width)
        time2 = time.time()
        #print('time', time2 - time1)
    except RuntimeError as e:
        print('interpolate RunTimeError')
        if upscale == 1:
            zpred = heatmap
        else:
            zpred = F.interpolate(
                heatmap.unsqueeze(0), scale_factor=upscale, mode="bilinear", align_corners=False
            ).squeeze(0)
    return zpred


def gaussian_sample(heatmap, point_coord):
    try:
        np_heatmap = heatmap.clone().detach().cpu().numpy()
        params = gaussian_param(np_heatmap)
        xy = point_coord.clone().permute(1, 0).detach().cpu().numpy()
        zpred = torch.from_numpy(gauss2d(xy, *params)).type(dtype)
    except RuntimeError as e:
        zpred = F.grid_sample(heatmap[None, None, :, :], 2 * point_coord[None, None, :, :] - 1.0).squeeze()
    return zpred


def generate_M(xy, x0, y0, height, width):
    x, y = xy
    M = np.zeros((height * width, 2))
    M[:, 0] = np.power(x - x0, 2) + np.power(y - y0, 2)
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
