# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random

import numpy as np
import torchvision
import cv2
from PIL import Image
import io

from core.inference import get_max_preds
from matplotlib import pyplot as plt
import os
import matplotlib.patches as mpathes

from utils.heatmap_func import gaussian


def vis_single_bbox_and_sample_point(heatmaps, cat_bboxs, point_coords_wrt_heatmap):
    numpy_heatmaps = heatmaps.clone().detach().cpu().numpy()
    numpy_bboxs = cat_bboxs.clone().detach().cpu().numpy()
    numpy_coords = point_coords_wrt_heatmap.clone().detach().cpu().numpy()

    B, C, H, W = numpy_heatmaps.shape

    i = random.randint(0, B-1)
    j = random.randint(0, C-1)

    idx = i * C + j
    bbox = numpy_bboxs[idx]
    heatmap = numpy_heatmaps[i, j]
    coord = numpy_coords[j, i]

    min_x, min_y, max_x, max_y = bbox

    fig, ax = plt.subplots()
    rect = mpathes.Rectangle((min_x, min_y), max_x - min_x,
                             max_y - min_y, color='r', fill=False)

    plt.scatter(
        coord[:, 0], coord[:, 1], c='red', marker='o', s=1
    )

    ax.add_patch(rect)
    plt.imshow(heatmap)
    plt.colorbar()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    im = Image.open(buf)
    #buf.close()
    return im


def vis_stage_heatmaps(stage_heatmaps, gaussian_heatmap_params):
    stage_num = len(stage_heatmaps)

    stage_ims = []

    for i in range(stage_num):
        heatmap = stage_heatmaps[i].clone().detach().cpu().numpy()

        gaussian_heatmap= gaussian_heatmap_params[i]
        gaussian_heatmap = gaussian_heatmap.clone().detach().cpu().numpy()

        plt.matshow(heatmap, cmap=plt.cm.gist_earth_r)
        plt.contour(gaussian_heatmap, cmap=plt.cm.copper)
        ax = plt.gca()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        im = Image.open(buf)
        #buf.close()
        stage_ims.append(im)

    return stage_ims


def vis_bbox_and_sample_point(heatmaps, cat_bboxs, point_coords_wrt_heatmap):
    numpy_heatmaps = heatmaps.clone().detach().cpu().numpy()
    numpy_bboxs = cat_bboxs.clone().detach().cpu().numpy()
    numpy_coords = point_coords_wrt_heatmap.clone().detach().cpu().numpy()

    B, C, H, W = numpy_heatmaps.shape

    for i in range(B):
        for j in range(C):
            idx = i * C + j
            bbox = numpy_bboxs[idx]
            heatmap = numpy_heatmaps[i, j]
            coord = numpy_coords[j, i]

            min_x, min_y, max_x, max_y = bbox

            fig, ax = plt.subplots()
            rect = mpathes.Rectangle((min_x, min_y), max_x - min_x,
                                     max_y - min_y, color='r', fill=False)

            plt.scatter(
                coord[:, 0], coord[:, 1], c='red', marker='o', s=1
            )

            ax.add_patch(rect)
            plt.imshow(heatmap)
            plt.colorbar()
            plt.show()

    #
    # for bbox in cat_boxes:
    #     numpy_bbox = bbox.clone().detach().cpu().numpy()[0]
    #



def save_val_debug_heatmaps(output_dict, prefix):
    coarse_heatmaps = output_dict['coarse']
    batch_size = coarse_heatmaps.size(0)
    num_joints = coarse_heatmaps.size(1)
    heatmap_height = coarse_heatmaps.size(2)
    heatmap_width = coarse_heatmaps.size(3)

    refine_heatmaps = output_dict['refine']
    stage_heatmaps = output_dict['stage_heatmaps']
    stage_point_indices = output_dict['stage_point_indices']

    coarse_heatmaps = coarse_heatmaps.detach().cpu().numpy()
    refine_heatmaps = refine_heatmaps.detach().cpu().numpy()

    for i in range(batch_size):
        individual_root = os.path.join(prefix, '{}'.format(i))
        os.makedirs(individual_root, exist_ok=True)
        coarse_heatmap = coarse_heatmaps[i]
        refine_heatmap = refine_heatmaps[i]

        for j in range(1):
            joint_root = os.path.join(individual_root, '{}'.format(j))
            os.makedirs(joint_root, exist_ok=True)
            c_heatmap = coarse_heatmap[j]
            r_heatmap = refine_heatmap[j]

            plt.imshow(c_heatmap, interpolation='none')
            plt.colorbar()
            plt.title('coarse')
            plt.savefig(os.path.join(joint_root, 'coarse.svg'), format='svg', dpi=600)
            plt.close()

            plt.imshow(r_heatmap, interpolation='none')
            plt.colorbar()
            plt.title('refine')
            plt.savefig(os.path.join(joint_root, 'refine.svg'), format='svg', dpi=600)
            plt.close()

    for stage_num, stage_heatmap in enumerate(stage_heatmaps):
        width = heatmap_width * (2 ** (stage_num + 1))
        stage_point_indice = stage_point_indices[stage_num].detach().cpu().numpy()
        stage_heatmap = stage_heatmap.detach().cpu().numpy()
        print('point_indice', stage_point_indice.shape)
        print('stage{}_heatmap'.format(stage_num), stage_heatmap.shape)

        for i in range(batch_size):
            individual_root = os.path.join(prefix, '{}'.format(i))

            for j in range(1):
                joint_root = os.path.join(individual_root, '{}'.format(j))
                os.makedirs(joint_root, exist_ok=True)
                s_heatmap = stage_heatmap[i][j]
                s_point_indice = stage_point_indice[i][j]

                stage_point_indice_x = [indice % width for indice in s_point_indice]
                stage_point_indice_y = [indice // width for indice in s_point_indice]

                plt.imshow(s_heatmap)
                plt.colorbar()
                plt.scatter(stage_point_indice_x, stage_point_indice_y, c='red', marker='o', s=0.5)
                plt.savefig(os.path.join(joint_root, 'stage_{}.svg'.format(stage_num)), format='svg', dpi=600)
                plt.close()

def save_batch_heatmaps_with_point_indices(batch_heatmaps, batch_point_indices, file_name):
    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    root = os.path.join(file_name, 'point_indices')
    os.makedirs(root, exist_ok=True)

    for i in range(batch_size):
        heatmaps = batch_heatmaps[i]
        point_indices = batch_point_indices[i]

        for j in range(num_joints):
            heatmap = heatmaps[j, :, :]
            point_indice = point_indices[j]

            stage_point_indice_x = [indice % heatmap_width for indice in point_indice]
            stage_point_indice_y = [indice // heatmap_width for indice in point_indice]

            plt.matshow(heatmap)
            plt.scatter(stage_point_indice_x, stage_point_indice_y, c='red', marker='o', s=1)
            plt.savefig('{}_{}_{}.svg'.format(file_name, i, j), format='svg', dpi=600)
            plt.close()

def save_batch_heatmaps_with_point_coords(batch_heatmaps, batch_point_coords, file_name):
    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    root = os.path.join(file_name, 'point_coords')
    os.makedirs(root, exist_ok=True)

    for i in range(batch_size):
        heatmaps = batch_heatmaps[i]
        point_coords = batch_point_coords[i]

        for j in range(num_joints):
            heatmap = heatmaps[j, :, :]
            point_coord = point_coords[j]

            stage_point_indice_x = point_coord[1] * heatmap_width
            stage_point_indice_y = point_coord[0] * heatmap_height

            plt.matshow(heatmap)
            plt.scatter(stage_point_indice_x, stage_point_indice_y, c='red', marker='o', s=1)
            plt.savefig(os.path.join(root, '{}_{}_{}.svg'.format(file_name, i, j)), format='svg', dpi=600)
            plt.close()


def save_batch_heatmaps(batch_image, batch_heatmaps, normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    return grid_image

def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
            k = k + 1
    return ndarr


def save_debug_images(config, input, meta, target, joints_pred, output):
    if not config.DEBUG.DEBUG:
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        gt_image = save_batch_image_with_joints(
            input, meta['joints'], meta['joints_vis']
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        pred_image = save_batch_image_with_joints(
            input, joints_pred, meta['joints_vis']
        )
    if config.DEBUG.SAVE_HEATMAPS_GT:
        hm_gt_image = save_batch_heatmaps(
            input, target
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        hm_pred_image = save_batch_heatmaps(
            input, output
        )

    return gt_image, pred_image, hm_gt_image, hm_pred_image
