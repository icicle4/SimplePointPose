import numpy as np
import torch
import torch.nn.functional as F


def calculate_uncertain_gaussian_heatmap_func(heatamp, upscale=1):
    if len(heatamp.size()) == 3:
        heatamp = torch.squeeze(heatamp, dim=0)

    if upscale == 1:
        #print('heatmap', heatamp.size(), heatamp.type(), heatamp.device)
        gaussian_heatmap = gaussian_interpolate(heatamp, 1)
        row_heatmap = heatamp
        #print('row',row_heatmap.device, row_heatmap.type())
        #print('gaussian',gaussian_heatmap.device, gaussian_heatmap.type())
        diff_map = torch.abs(row_heatmap - gaussian_heatmap)
    else:
        gaussian_heatmap = gaussian_interpolate(heatamp, upscale)
        interpolated_heatmap = F.interpolate(
            heatamp, scale_factor=upscale, mode="bilinear", align_corners=False
        ).squeeze(0)
        diff_map = torch.abs(interpolated_heatmap - gaussian_heatmap)
    return diff_map


def gaussian_interpolate(heatmap, upsample_scale):
    params = moment_torch(heatmap)
    fit = gaussian_torch(*params)
    indices = torch.from_numpy(np.indices(np.array(heatmap.shape) * upsample_scale) / upsample_scale).float().cuda()
    new_heatmap = fit(*indices)
    return new_heatmap


def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)


def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y


def gaussian_torch(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = width_x.float()
    width_y = width_y.float()
    return lambda x,y: height*torch.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)


def moment_torch(data):
    height, width = data.size()
    #print('height, width', height, width)
    total = torch.sum(data)
    X, Y = np.indices(data.shape)
    X = torch.from_numpy(X).cuda()
    Y = torch.from_numpy(Y).cuda()
    x = torch.sum((X * data)) / total
    y = torch.sum((Y * data)) / total
    #print('x, y', x, y)
    col = data[:, torch.clamp(y, -width+1, width-1).long()]
    width_x = torch.sqrt(torch.sum(torch.abs(torch.arange(col.size()[0]).cuda() - y)**2 * col) / torch.sum(col))
    row = data[torch.clamp(x, -height+1, height-1).long(), :]
    width_y = torch.sqrt(torch.sum(torch.abs(torch.arange(row.size()[0]).cuda() - x)**2 * row) / torch.sum(row))
    height = torch.max(data)
    return height, x, y, width_x, width_y

