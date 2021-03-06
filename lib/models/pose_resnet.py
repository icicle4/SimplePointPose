# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from collections import OrderedDict
from utils.point_func import *
import fvcore.nn.weight_init as weight_init
from utils.heatmap_func import *
from utils.vis import vis_bbox_and_sample_point



BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck_CAFFE(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_CAFFE, self).__init__()
        # add stride to conv1x1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PointHead(nn.Module):
    def __init__(self, fc_dim, num_fc):
        super(PointHead, self).__init__()
        self.num_classes = 1
        self.input_channels = fc_dim
        self.num_fc = num_fc

        fc_dim_in = self.input_channels + self.num_classes
        self.fc_layers = nn.ModuleList()
        layers = list()
        for k in range(num_fc):
            fc = nn.Conv1d(fc_dim_in, fc_dim, kernel_size=1, stride=1, padding=0, bias=True)
            layers.append(fc)
            fc_dim_in = fc_dim
            fc_dim_in += 1
        self.fc_layers = nn.ModuleList(layers)
        self.predictor = nn.Conv1d(fc_dim_in, 1, kernel_size=1, stride=1, padding=0)

        for i, layer in enumerate(self.fc_layers):
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self, fine_grained_features, fine_deconv_features, coarse_features):
        x = torch.cat((fine_grained_features, fine_deconv_features, coarse_features), dim=1)
        for i, layer in enumerate(self.fc_layers):
            x = F.relu(layer(x))
            x = torch.cat((x, coarse_features), dim=1)
        return self.predictor(x)


class PoseResNet(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.point_head = PointHead(256 + 256, 3)

        self.train_num_points = 7 * 7
        self.oversample_ratio = 3.0
        self.importance_sample_ratio = 0.75

        self.subdivision_steps = 3
        self.subdivision_num_points = 14 * 14

        self.freeze_gaussian = True

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x, gt_heatmaps):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        backbone = self.layer1(x)

        x = self.layer2(backbone)
        x = self.layer3(x)
        x = self.layer4(x)

        deconv_features = self.deconv_layers(x)
        coarse_heatmaps = self.final_layer(deconv_features)

        if self.training:
            heatmap_loss = (F.mse_loss(coarse_heatmaps, gt_heatmaps, reduction='mean'))

            B, C, H, W = coarse_heatmaps.size()

            with torch.no_grad():
                proposal_boxes, point_coords = get_certain_point_coors_with_randomness(
                    coarse_heatmaps,
                    lambda logits: calculate_certainty(logits, []),
                    self.train_num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio)

                _, num_sampled, _ = point_coords.size()

            cat_boxes = torch.cat(proposal_boxes, dim=0)
            point_coords_wrt_heatmap = get_point_coords_wrt_roi(cat_boxes, point_coords)
            point_coords_wrt_heatmap = point_coords_wrt_heatmap.view(B, C, num_sampled, 2).permute(1, 0, 2, 3)

            #vis_bbox_and_sample_point(coarse_heatmaps, cat_boxes, point_coords_wrt_heatmap)

            fine_grained_backbone = point_sample_fine_grained_features(
                backbone, point_coords_wrt_heatmap
            )

            fine_grained_deconv = point_sample_fine_grained_features(
                deconv_features, point_coords_wrt_heatmap
            )

            coarse_features = point_sample_pd_heatmaps(
                coarse_heatmaps, point_coords_wrt_heatmap
            )
            point_logits = self.point_head(fine_grained_backbone, fine_grained_deconv, coarse_features)

            # 此处我们将interpolate插值替换为gaussian插值
            if self.freeze_gaussian:
                with torch.no_grad():
                    gt_point_logits = []
                    D, C, H, W = gt_heatmaps.shape
                    for i in range(D):
                        for j in range(C):
                            point_coord_wrt_heatmap = point_coords_wrt_heatmap[j, i]
                            point_coord_wrt_heatmap = point_coord_wrt_heatmap.permute(1, 0)

                            heatmap = gt_heatmaps[i, j]
                            if torch.sum(heatmap) ==0:
                                params = [
                                    torch.tensor(0.00001).cuda(),
                                    torch.tensor(24).cuda(),
                                    torch.tensor(32).cuda(),
                                    torch.tensor(72).cuda(),
                                    torch.tensor(96).cuda(),
                                ]
                            else:
                                params = moment_torch(heatmap)

                            if sum([torch.isnan(p) for p in params])>0:
                                params = [
                                    torch.tensor(0.00001).cuda(),
                                    torch.tensor(24).cuda(),
                                    torch.tensor(32).cuda(),
                                    torch.tensor(72).cuda(),
                                    torch.tensor(96).cuda(),
                                ]

                            fit = gaussian_torch(*params)
                            gt_point_logit = fit(*point_coord_wrt_heatmap)
                            gt_point_logits.append(gt_point_logit[None, None, :])
                    gt_point_logits = torch.cat(gt_point_logits)
            else:
                gt_point_logits = []
                D, C, H, W = gt_heatmaps.shape
                for i in range(D):
                    for j in range(C):
                        point_coord_wrt_heatmap = point_coords_wrt_heatmap[j, i]
                        point_coord_wrt_heatmap = point_coord_wrt_heatmap.permute(1, 0)

                        heatmap = gt_heatmaps[i, j]
                        params = moment_torch(heatmap)
                        fit = gaussian_torch(*params)
                        gt_point_logit = fit(*point_coord_wrt_heatmap)

                        gt_point_logits.append(gt_point_logit)
                gt_point_logits = torch.cat(gt_point_logits)

            point_loss = F.mse_loss(point_logits, gt_point_logits, reduction='mean')
            loss = heatmap_loss + point_loss

            output_heatmaps = coarse_heatmaps.clone()

            return {
                "output": output_heatmaps,
                "loss": loss,
                "heatmap_loss": heatmap_loss,
                "point_loss": point_loss,
                'point_coords': point_coords
            }

        else:

            heatmaps_logits = coarse_heatmaps.clone()
            D, C, H, W = heatmaps_logits.size()

            stage_heatmaps = []
            stage_point_indices = []

            for subdivision_step in range(self.subdivision_steps):
                upsampled_heatmap_logits = []
                point_coords = []
                point_indices = []

                for i in range(D):
                    for j in range(C):
                        heatmap_logit = heatmaps_logits[i, j]
                        # 每一阶段，初始heatmap将分别计算高斯分布参数生成更高分辨率的特征图及通过线性插值生成更高分辨率的特征图
                        # 我们将两种特征图中差异最大的值作为我们需要优化的目标
                        gaussian_heatmap = gaussian_interpolate(heatmap_logit, 2)
                        interpolated_heatmap = F.interpolate(
                            heatmap_logit[None, None, :, :], scale_factor=2, mode="bilinear", align_corners=False
                        ).squeeze(0)

                        error_map = torch.abs(gaussian_heatmap - interpolated_heatmap)
                        uncertain_map = calculate_certainty(error_map.unsqueeze(dim=0), [])

                        point_indice, point_coord = get_certain_point_coords_on_grid(
                            uncertain_map, num_points=self.subdivision_num_points
                        )

                        point_indices.append(point_indice)
                        point_coords.append(point_coord)

                        upsampled_heatmap_logits.append(interpolated_heatmap)

                point_indices = torch.cat(point_indices)
                point_coords = torch.cat(point_coords).view(D, C, -1, 2).permute(1, 0, 2, 3)

                fine_grained_backbone = point_sample_features(backbone, point_coords)
                fine_grained_deconv = point_sample_features(deconv_features, point_coords)
                coarse_features = point_sample_pd_features(coarse_heatmaps, point_coords)

                upsampled_heatmap_logits = torch.cat(upsampled_heatmap_logits, dim=0)
                _, new_H, new_W = upsampled_heatmap_logits.size()
                heatmaps_logits = upsampled_heatmap_logits.view(D, C, new_H, new_W)
                stage_heatmaps.append(heatmaps_logits)

                point_logits = self.point_head(fine_grained_backbone, fine_grained_deconv, coarse_features)
                point_logits = point_logits.squeeze(1)
                # put mask point predictions to the right places on the upsampled grid.
                R, C, H, W = heatmaps_logits.shape
                heatmaps_logits = (
                    heatmaps_logits.reshape(R * C, H * W)
                        .scatter_(1, point_indices, point_logits)
                        .view(R, C, H, W)
                )
                stage_point_indices.append(point_indices.view(D, C, -1))
            return {
                'refine': heatmaps_logits,
                'coarse': coarse_heatmaps,
                'stage_heatmaps': stage_heatmaps,
                'stage_point_indices': stage_point_indices
            }


    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            # pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            # self.load_state_dict(pretrained_state_dict, strict=False)
            checkpoint = torch.load(pretrained)
            if isinstance(checkpoint, OrderedDict):
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict_old = checkpoint['state_dict']
                state_dict = OrderedDict()
                # delete 'module.' because it is saved from DataParallel module
                for key in state_dict_old.keys():
                    if key.startswith('module.'):
                        # state_dict[key[7:]] = state_dict[key]
                        # state_dict.pop(key)
                        state_dict[key[7:]] = state_dict_old[key]
                    else:
                        state_dict[key] = state_dict_old[key]
            else:
                raise RuntimeError(
                    'No state_dict found in checkpoint file {}'.format(pretrained))
            self.load_state_dict(state_dict, strict=False)
        else:
            logger.error('=> imagenet pretrained model dose not exist')
            logger.error('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    style = cfg.MODEL.STYLE

    block_class, layers = resnet_spec[num_layers]

    if style == 'caffe':
        block_class = Bottleneck_CAFFE

    model = PoseResNet(block_class, layers, cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model