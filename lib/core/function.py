# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os

import numpy as np
import torch

from core.config import get_model_name
from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images, vis_single_bbox_and_sample_point, vis_stage_heatmaps

import wandb


logger = logging.getLogger(__name__)


def train(config, train_loader, model, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    heatmap_losses = AverageMeter()
    point_losses = AverageMeter()
    losses = AverageMeter()

    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    TrainBboxSamplePoints = []
    TrainGtImages = []
    TrainGtHeatmaps = []
    TrainPdImages = []
    TrainPdHeatmaps = []

    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        group_id = epoch * len(train_loader) + i
        data_time.update(time.time() - end)
        # compute output
        output_dict = model(input, target)
        loss = output_dict['loss']
        heatmap_loss = output_dict['heatmap_loss']
        point_loss = output_dict['point_loss']

        coarse_heatmaps = output_dict['output']
        cat_boxes = output_dict['cat_boxes']
        point_coords_wrt_heatmap = output_dict['point_coords_wrt_heatmap']

        target = target.cuda(non_blocking=True)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        heatmap_losses.update(heatmap_loss.item(), input.size(0))
        point_losses.update(point_loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output_dict['output'].detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'PointLoss {point_loss.val:.5f} ({point_loss.avg:.5f})\t' \
                  'HeatmapLoss {heatmap_loss.val:.5f} ({heatmap_loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses,
                    point_loss= point_losses,
                    heatmap_loss = heatmap_losses,
                    acc=acc)
            logger.info(msg)
            global_steps = writer_dict['train_global_steps']

            TrainBboxSamplePoints.append(
                wandb.Image(
                    vis_single_bbox_and_sample_point(coarse_heatmaps, cat_boxes, point_coords_wrt_heatmap),
                    caption='Train Epoch {} bbox and sample point'.format(epoch),
                    grouping=group_id
                )
            )

            gt_image, pred_image, hm_gt_image, hm_pred_image = save_debug_images(config,
                                                                                 input,
                                                                                 meta, target,
                                                                                 pred * 4, output_dict['output'])
            TrainGtHeatmaps.append(
                wandb.Image(
                    hm_gt_image, caption='Train Gt heatmap', grouping=group_id
                )
            )
            TrainGtImages.append(
                wandb.Image(
                    gt_image, caption='Train Gt Images', grouping=group_id
                )
            )
            TrainPdHeatmaps.append(
                wandb.Image(
                    hm_pred_image, caption='Train Gt heatmap', grouping=group_id
                )
            )
            TrainPdImages.append(
                wandb.Image(
                    pred_image, caption='Train Pred heatmap', grouping=group_id
                )
            )

    wandb.log(
        {
            'Train Loss': losses.avg,
            'Train Point Loss': point_losses.avg,
            'Train Heatmap Loss': heatmap_losses.avg,
            'Train Accuracy': acc.avg,
            'Train BBox SamplePoints': TrainBboxSamplePoints,
            'Train Gt Images': TrainGtImages,
            'Train Pd Images': TrainPdImages,
            'Train Gt Heatmaps': TrainGtHeatmaps,
            'Train Pd Heatmaps': TrainPdHeatmaps
        }
    )


def validate(config, val_loader, val_dataset, model, output_dir,
             tb_log_dir, epoch):
    batch_time = AverageMeter()
    coarse_acc = AverageMeter()
    refine_acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3),
                         dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0

    ValCoarseGtImages = []
    ValCoarseGtHeatmaps = []
    ValCoarsePdImages = []
    ValCoarsePdHeatmaps = []

    ValRefineGtImages = []
    ValRefineGtHeatmaps = []
    ValRefinePdImages = []
    ValRefinePdHeatmaps = []

    StageImages1 = []
    StageImages2 = []
    StageImages3 = []

    StageRefineHeatmaps1 = []
    StageRefineHeatmaps2 = []
    StageRefineHeatmaps3 = []

    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            group_id = epoch * len(val_loader) + i
            output_dict = model(input, target)
            refine_output, coarse_output = output_dict['refine'], output_dict['coarse']

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()

                refine_output_flipped, coarse_output_flipped = model(input_flipped, target)

                refine_output_flipped = flip_back(refine_output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                refine_output_flipped = torch.from_numpy(refine_output_flipped.copy()).cuda()

                coarse_output_flipped = flip_back(coarse_output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                coarse_output_flipped = torch.from_numpy(coarse_output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    coarse_output_flipped[:, :, :, 1:] = \
                        coarse_output_flipped.clone()[:, :, :, 0:-1]
                    refine_output_flipped[:, :, :, 1:] = \
                        refine_output_flipped.clone()[:, :, :, 0:-1]
                    # output_flipped[:, :, :, 0] = 0

                coarse_output = (coarse_output + coarse_output_flipped) * 0.5
                refine_output = (refine_output + refine_output_flipped) * 0.5

            target = target.cuda(non_blocking=True)

            num_images = input.size(0)
            # measure accuracy and record loss

            _, coarse_avg_acc, coarse_cnt, coarse_pred = accuracy(coarse_output.cpu().numpy(),
                                             target.cpu().numpy())

            _, refine_avg_acc, refine_cnt, refine_pred = accuracy(refine_output.cpu().numpy(),
                                                                  target.cpu().numpy())

            coarse_acc.update(coarse_avg_acc, coarse_cnt)
            refine_acc.update(refine_avg_acc, refine_cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(config, refine_output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])
            if config.DATASET.DATASET == 'posetrack':
                filenames.extend(meta['filename'])
                imgnums.extend(meta['imgnum'].numpy())

            idx += num_images

            if i % config.PRINT_FREQ == 0 or i == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Coarse Accuracy {coarse_acc.val:.3f} ({coarse_acc.avg:.3f})\t' \
                      'Refine Accuracy {refine_acc.val:.3f} ({refine_acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          coarse_acc=coarse_acc, refine_acc=refine_acc)
                logger.info(msg)

                #stage_point_indices = output_dict['stage_point_indices']
                stage_gaussian_params = output_dict['stage_gaussian_params']
                stage_interpolate_heatmaps = output_dict['stage_interpolate_heatmaps']
                stage_refined_heatmaps = output_dict['stage_refined_heatmaps']

                coarse_gt_image, coarse_pred_image, coarse_hm_gt_image, coarse_hm_pred_image = save_debug_images(config,
                                                                                     input,
                                                                                     meta, target,
                                                                                     coarse_pred * 4, coarse_output)

                refine_gt_image, refine_pred_image, refine_hm_gt_image, refine_hm_pred_image = save_debug_images(config,
                                                                                     input,
                                                                                     meta, target,
                                                                                     coarse_pred * 4, coarse_output)

                stage_ims = vis_stage_heatmaps(stage_interpolate_heatmaps, stage_gaussian_params)
                stage_refined_ims = vis_stage_heatmaps(stage_refined_heatmaps, stage_gaussian_params)

                StageImages1.append(
                    wandb.Image(
                        stage_ims[0], caption='Stage Image 1', grouping=group_id
                    )
                )
                StageImages2.append(
                    wandb.Image(
                        stage_ims[1], caption='Stage Image 2', grouping=group_id
                    )
                )
                StageImages3.append(
                    wandb.Image(
                        stage_ims[2], caption='Stage Image 3', grouping=group_id
                    )
                )

                StageRefineHeatmaps1.append(
                    wandb.Image(
                        stage_refined_ims[0], caption='Stage Refine Heatmaps 1', grouping=group_id
                    )
                )

                StageRefineHeatmaps2.append(
                    wandb.Image(
                        stage_refined_ims[1], caption='Stage Refine Heatmaps 2', grouping=group_id
                    )
                )

                StageRefineHeatmaps3.append(
                    wandb.Image(
                        stage_refined_ims[2], caption='Stage Refine Heatmaps 3', grouping=group_id
                    )
                )

                ValCoarseGtImages.append(
                    wandb.Image(
                        coarse_gt_image, caption='Val Coarse Gt Image', grouping=group_id
                    )
                )
                
                ValRefineGtImages.append(
                    wandb.Image(
                        refine_gt_image, caption='Val Refine Gt Image', grouping=group_id
                    )
                )
                
                ValCoarseGtHeatmaps.append(
                    wandb.Image(
                        coarse_hm_gt_image, caption='Val Coarse Gt Heatmap', grouping=group_id
                    )
                )
                
                ValCoarseGtHeatmaps.append(
                    wandb.Image(
                        refine_hm_gt_image, caption='Val Refine Gt Heatmap', grouping=group_id
                    )
                )
                
                ValCoarsePdImages.append(
                    wandb.Image(
                        coarse_pred_image, caption='Val Coarse Pd Image', grouping=group_id
                    )
                )
                
                ValRefineGtImages.append(
                    wandb.Image(
                        refine_pred_image, caption='Val Refine Pd Image', grouping=group_id
                    )
                )
                
                ValCoarsePdHeatmaps.append(
                    wandb.Image(
                        coarse_hm_pred_image, caption='Val Coarse Pd Heatmap', grouping=group_id
                    )
                )
                
                ValRefineGtHeatmaps.append(
                    wandb.Image(
                        refine_hm_pred_image, caption='Val Refine Pd heatmaps', grouping=group_id
                    )
                )


        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums)

        _, full_arch_name = get_model_name(config)
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, full_arch_name)
        else:
            _print_name_value(name_values, full_arch_name)

    wandb.log(
        {
            'Val coarse acc': coarse_acc.avg,
            'Val refine acc': refine_acc.avg,
            'Val Coarse Gt Images': ValCoarseGtImages,
            'Val Coarse Pd Images': ValCoarsePdImages,
            'Val Coarse Gt Heatmaps': ValCoarseGtHeatmaps,
            'Val Coarse Pd Heatmaps': ValCoarsePdHeatmaps,
            'Val Refine Gt Images': ValRefineGtImages,
            'Val Refine Pd Images': ValRefinePdImages,
            'Val Refine Gt Heatmaps': ValRefineGtHeatmaps,
            'Val Refine Pd Heatmaps': ValRefinePdHeatmaps,
            'Val Stage Heatmap 1': StageImages1,
            'Val Stage Heatmap 2': StageImages2,
            'Val Stage Heatmap 3': StageImages3,
            'Val Stage Refine Heatmap 1': StageRefineHeatmaps1,
            'Val Stage Refine Heatmap 2': StageRefineHeatmaps2,
            'Val Stage Refine Heatmap 3': StageRefineHeatmaps3
        }
    )
    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
