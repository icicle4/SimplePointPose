# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from utils.transforms import transform_preds
import torch


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i],
                                   [heatmap_width, heatmap_height])

    return preds, maxvals

def _box2cs(box):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h)

def _xywh2cs(x, y, w, h):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > h:
        h = w * 1.0
    elif w < h:
        w = h
    scale = np.array(
        [w * 1.0 / 200., h * 1.0 / 200.],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25
    return center, scale

# if __name__ == '__main__':
#     box = [10, 20, 10, 10]
#     center, scale = _box2cs(box)
#     print(center, scale)
#     small_heatmaps = np.zeros((1, 1, 64, 48))
#     small_heatmaps[:, :, 10, 20] = 1
#     preds, maxvals = get_final_preds(None, small_heatmaps, center, scale)
#     print('small', preds, maxvals)
#
#     large_heatmaps = np.zeros((1, 1, 64*8, 48*8))
#     large_heatmaps[:, :, 80, 160] = 1
#     preds, maxvals = get_final_preds(None, small_heatmaps, center, scale)
#     print('large', preds, maxvals)
