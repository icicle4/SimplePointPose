import numpy as np
import torch


def gaussian_interpolate(heatmap, upsample_scale):
    params = moment_torch(heatmap)
    fit = gaussian_torch(*params)
    indices = torch.from_numpy(np.indices(np.array(heatmap.shape) * upsample_scale) / upsample_scale)
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
    width_y = np.sqrt(
        np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum()
    )
    height = data.max()
    return height, x, y, width_x, width_y


def gaussian_torch(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = width_x.float()
    width_y = width_y.float()
    return lambda x,y: height*torch.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)


def moment_torch(data):
    total = torch.sum(data)
    X, Y = np.indices(data.shape)
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)
    x = torch.sum((X * data)) / total
    y = torch.sum((Y * data)) / total
    col = data[:, y.long()]
    width_x = torch.sqrt(torch.sum(torch.abs(torch.arange(col.size) - y)**2 * col) / torch.sum(col))
    row = data[x.long(), :]
    width_y = torch.sqrt(torch.sum(torch.abs(torch.arange(row.size) - y)**2 * row) / torch.sum(row))
    height = torch.max(data)
    return height, x, y, width_x, width_y

