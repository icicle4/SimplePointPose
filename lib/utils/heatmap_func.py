import numpy as np
import torch
import torch.nn.functional as F
import scipy.optimize as opt


def simple_gauss2d(xy, amp=None, x0=None, y0=None, a=0, c=0):
    x, y = xy
    inner = a * (x - x0) ** 2
    inner += 2 * a * (x - x0) ** 2 * (y - y0) ** 2
    inner += c * (y - y0) ** 2
    return amp * np.exp(-inner)


def cal_gauss_param(heatmap):
    height, width = heatmap.shape
    xi = np.linspace(0, 1, width + 1)[:-1] + .5 / width
    yi = np.linspace(0, 1, height + 1)[:-1] + .5 / height

    Ys = yi.flatten()
    Xs = xi.flatten()
    zobs = heatmap.flatten()
    xy = np.concatenate([Xs[None, :], Ys[None, :]], axis=0)
    x, y = xy
    i = zobs.argmax()
    guess = [1, 1]
    simply_gauss = lambda xy, a, c: simple_gauss2d(xy, amp=zobs.max(), x0=x[i], y0=y[i], a=a, c=c)
    pred_params, uncert_cov = opt.curve_fit(simply_gauss, xy, zobs, p0=guess)
    return pred_params, simply_gauss


def gaussian_interpolate(heatmap, upscale=1):
    np_heatmap = heatmap.clone().detach().cpu().numpy()
    height, width = heatmap.shape

    params, gauss_func = cal_gauss_param(np_heatmap)

    up_height, up_width = height * upscale, width * upscale

    xi = np.linspace(0, 1, up_width + 1)[:-1] + .5 / up_width
    yi = np.linspace(0, 1, up_height + 1)[:-1] + .5 / up_height

    Ys = yi.flatten()
    Xs = xi.flatten()
    xy = np.concatenate([Xs[None, :], Ys[None, :]], axis=0)

    zpred = gauss_func(xy, *params)
    return torch.from_numpy(zpred).cuda().view(up_height, up_width)


def gaussian_sample(heatmap, point_coord):
    np_heatmap = heatmap.clone().detach().cpu().numpy()
    params, gauss_func = cal_gauss_param(np_heatmap)
    xy = point_coord.clone().detach().cpu().numpy()
    zpred = gauss_func(xy, *params).cuda()
    return zpred


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
