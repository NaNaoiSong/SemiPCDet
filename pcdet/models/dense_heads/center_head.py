import torch
import torch.nn as nn
import copy
import sys
from collections import OrderedDict
import numpy as np
import numba
from ...ops.iou3d_nms import iou3d_nms_cuda
from ...ops.center_ops import center_ops_cuda

from ...utils import loss_utils
from .target_assigner.center_assigner import CenterAssigner

class CenterHead(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.in_channels = input_channels
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.out_size_factor = model_cfg.TARGET_ASSIGNER_CONFIG.out_size_factor
        self.predict_boxes_when_training = predict_boxes_when_training

        self.num_classes = [len(t["class_names"]) for t in model_cfg.TASKS]
        self.class_names = [t["class_names"] for t in model_cfg.TASKS]

        self.code_weights = model_cfg.LOSS_CONFIG.code_weights
        self.weight = model_cfg.LOSS_CONFIG.weight # weight between hm loss and loc loss

        self.no_log = False

        # a shared convolution
        share_conv_channel = model_cfg.PARAMETERS.share_conv_channel
        self.shared_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, share_conv_channel,
                      kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(share_conv_channel),
            nn.ReLU(inplace=True)
        )

        self.common_heads = model_cfg.PARAMETERS.common_heads
        self.init_bias = model_cfg.PARAMETERS.init_bias
        self.tasks = nn.ModuleList()

        for num_cls in self.num_classes:
            heads = copy.deepcopy(self.common_heads)
            heads.update(dict(hm=(num_cls, 2)))
            self.tasks.append(
                SepHead(share_conv_channel, heads, bn=True, init_bias=self.init_bias, final_kernel=3, directional_classifier=False)
            )
        self.target_assigner = CenterAssigner(
            model_cfg.TASKS,
            model_cfg.TARGET_ASSIGNER_CONFIG,
            num_classes = sum(self.num_classes),
            no_log = self.no_log,
            grid_size = grid_size,
            pc_range = point_cloud_range,
            voxel_size = voxel_size
        )

        self.forward_ret_dict = {}
        self.build_losses()
        self.model_type = 'origin'

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets_v2(
            gt_boxes
        )

        return targets_dict

    def forward(self, data_dict):
        multi_head_features = []
        spatial_features_2d = data_dict['spatial_features_2d']
        spatial_features_2d = self.shared_conv(spatial_features_2d)
        for task in self.tasks:
            multi_head_features.append(task(spatial_features_2d))

        self.forward_ret_dict['multi_head_features'] = multi_head_features

        if self.model_type == 'origin':
            if self.training:
                targets_dict = self.assign_targets(
                    gt_boxes=data_dict['gt_boxes']
                )
                self.forward_ret_dict.update(targets_dict)

            if not self.training or self.predict_boxes_when_training:
                batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(data_dict)

                if isinstance(batch_cls_preds, list):
                    multihead_label_mapping = []
                    s = 1
                    for idx in range(len(self.num_classes)):
                        multihead_label_mapping.append(torch.arange(s, s + self.num_classes[idx]))
                        s += self.num_classes[idx]

                    data_dict['multihead_label_mapping'] = multihead_label_mapping

                data_dict['batch_cls_preds'] = batch_cls_preds
                data_dict['batch_box_preds'] = batch_box_preds
                data_dict['cls_preds_normalized'] = False
        elif self.model_type == 'teacher':
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(data_dict)

            if isinstance(batch_cls_preds, list):
                multihead_label_mapping = []
                s = 1
                for idx in range(len(self.num_classes)):
                    multihead_label_mapping.append(torch.arange(s, s + self.num_classes[idx]))
                    s += self.num_classes[idx]

                data_dict['multihead_label_mapping'] = multihead_label_mapping

            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        elif self.model_type == 'student':
            if self.training and 'gt_boxes' in data_dict:
                targets_dict = self.assign_targets(
                    gt_boxes=data_dict['gt_boxes']
                )
                self.forward_ret_dict.update(targets_dict)

            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(data_dict)

            if isinstance(batch_cls_preds, list):
                multihead_label_mapping = []
                s = 1
                for idx in range(len(self.num_classes)):
                    multihead_label_mapping.append(torch.arange(s, s + self.num_classes[idx]))
                    s += self.num_classes[idx]

                data_dict['multihead_label_mapping'] = multihead_label_mapping
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False
        else:
            raise NotImplementedError('Unsupported model type')

        return data_dict


    def build_losses(self):
        self.add_module(
            'crit',
            loss_utils.CenterNetFocalLoss()
        )
        self.add_module(
            'crit_reg',
            loss_utils.CenterNetRegLoss()
        )
        return

    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):
        tb_dict = {}
        pred_dicts = self.forward_ret_dict['multi_head_features']
        center_loss = []
        self.forward_ret_dict['pred_box_encoding'] = {}
        for task_id, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self._sigmoid(pred_dict['hm'])
            hm_loss = self.crit(pred_dict['hm'], self.forward_ret_dict['heatmap'][task_id])

            target_box_encoding = self.forward_ret_dict['box_encoding'][task_id]
            # nuscense encoding format [x, y, z, w, l, h, sinr, cosr, vx, vy]

            # if self.box_n_dim == 9:
            #     pred_box_encoding = torch.cat([
            #         pred_dict['reg'],
            #         pred_dict['height'],
            #         pred_dict['dim'],
            #         pred_dict['rot'],
            #         pred_dict['vel']
            #     ], dim = 1).contiguous() # (B, 10, H, W)
            # else:
            #     pred_box_encoding = torch.cat([
            #         pred_dict['reg'],
            #         pred_dict['height'],
            #         pred_dict['dim'],
            #         pred_dict['rot']
            #     ], dim = 1).contiguous() # (B, 8, H, W)

            pred_box_encoding = torch.cat([
                pred_dict['reg'],
                pred_dict['height'],
                pred_dict['dim'],
                pred_dict['rot']
            ], dim=1).contiguous()  # (B, 8, H, W)

            self.forward_ret_dict['pred_box_encoding'][task_id] = pred_box_encoding

            box_loss = self.crit_reg(
                pred_box_encoding,
                self.forward_ret_dict['mask'][task_id],
                self.forward_ret_dict['ind'][task_id],
                target_box_encoding
            )

            loc_loss = (box_loss * box_loss.new_tensor(self.code_weights)).sum()
            loss = hm_loss + self.weight * loc_loss

            tb_key = 'task_' + str(task_id) + '/'

            # if self.box_n_dim == 9:
            #     tb_dict.update({
            #         tb_key + 'loss': loss.item(), tb_key + 'hm_loss': hm_loss.item(), tb_key + 'loc_loss': loc_loss.item(),
            #         tb_key + 'x_loss': box_loss[0].item(), tb_key + 'y_loss': box_loss[1].item(), tb_key + 'z_loss': box_loss[2].item(),
            #         tb_key + 'w_loss': box_loss[3].item(), tb_key + 'l_loss': box_loss[4].item(), tb_key + 'h_loss': box_loss[5].item(),
            #         tb_key + 'sin_r_loss': box_loss[6].item(), tb_key + 'cos_r_loss': box_loss[7].item(),
            #         tb_key + 'vx_loss': box_loss[8].item(), tb_key + 'vy_loss': box_loss[9].item(),
            #         tb_key + 'num_positive': self.forward_ret_dict['mask'][task_id].float().sum(),
            #     })
            # else:
            #     tb_dict.update({
            #         tb_key + 'loss': loss.item(), tb_key + 'hm_loss': hm_loss.item(),
            #         tb_key + 'loc_loss': loc_loss.item(),
            #         tb_key + 'x_loss': box_loss[0].item(), tb_key + 'y_loss': box_loss[1].item(),
            #         tb_key + 'z_loss': box_loss[2].item(),
            #         tb_key + 'w_loss': box_loss[3].item(), tb_key + 'l_loss': box_loss[4].item(),
            #         tb_key + 'h_loss': box_loss[5].item(),
            #         tb_key + 'sin_r_loss': box_loss[6].item(), tb_key + 'cos_r_loss': box_loss[7].item(),
            #         tb_key + 'num_positive': self.forward_ret_dict['mask'][task_id].float().sum(),
            #     })

            tb_dict.update({
                tb_key + 'loss': loss.item(), tb_key + 'hm_loss': hm_loss.item(),
                tb_key + 'loc_loss': loc_loss.item(),
                tb_key + 'x_loss': box_loss[0].item(), tb_key + 'y_loss': box_loss[1].item(),
                tb_key + 'z_loss': box_loss[2].item(),
                tb_key + 'w_loss': box_loss[3].item(), tb_key + 'l_loss': box_loss[4].item(),
                tb_key + 'h_loss': box_loss[5].item(),
                tb_key + 'sin_r_loss': box_loss[6].item(), tb_key + 'cos_r_loss': box_loss[7].item(),
                tb_key + 'num_positive': self.forward_ret_dict['mask'][task_id].float().sum(),
            })
            center_loss.append(loss)

        return sum(center_loss), tb_dict

    @torch.no_grad()
    def generate_predicted_boxes(self, data_dict):
        """
        Generate box predictions with decode, topk and circular_nms
        For single-stage-detector, another post-processing (nms) is needed
        For two-stage-detector, no need for proposal layer in roi_head
        Returns:
        """
        pred_dicts = self.forward_ret_dict['multi_head_features']
        batch_box_preds = []
        batch_cls_preds = []

        for task_id, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['hm'].permute(0, 2, 3, 1)
            batch_reg = pred_dict['reg'].permute(0, 2, 3, 1)
            batch_hei = pred_dict['height'].permute(0, 2, 3, 1)

            if not self.no_log:
                batch_dim = torch.exp(pred_dict['dim'].permute(0, 2, 3, 1))
                # add clamp for good init, otherwise we will get inf with exp
                batch_dim = torch.clamp(batch_dim, min=0.001, max=30)
            else:
                batch_dim = pred_dict['dim'].permute(0, 2, 3, 1)
            batch_rots = pred_dict['rot'][:, 0].unsqueeze(1)
            batch_rotc = pred_dict['rot'][:, 1].unsqueeze(1)
            batch_rot = torch.atan2(batch_rots, batch_rotc).permute(0, 2, 3, 1)
            b, h, w, _ = batch_hm.shape
            inds = torch.arange(h * w).to(batch_hm.device)
            inds = inds.unsqueeze(0).repeat(b, 1)
            batch_ys = (inds // w).float().view(b, h, w, 1) + batch_reg[:, :, :, 1:2]
            batch_xs = (inds % w).float().view(b, h, w, 1) + batch_reg[:, :, :, 0:1]
            batch_xs = batch_xs * self.out_size_factor * self.voxel_size[0] + self.point_cloud_range[0]
            batch_ys = batch_ys * self.out_size_factor * self.voxel_size[1] + self.point_cloud_range[0]

            if pred_dict.get('vel', None) is not None:
                batch_vel = pred_dict['vel'].permute(0, 2, 3, 1)
                box_preds = torch.cat(
                    [batch_xs, batch_ys, batch_hei, batch_dim, batch_rot, batch_vel], dim=-1
                )
            else:
                box_preds = torch.cat(
                    [batch_xs, batch_ys, batch_hei, batch_dim, batch_rot], dim=-1
                )
            batch_cls_preds.append(batch_hm.view(b, w * h, -1))
            batch_box_preds.append(box_preds.view(b, w * h, -1))
        return batch_cls_preds, torch.cat(batch_box_preds, 1)

"""
BASIC BUILDING BLOCKS
"""
class Sequential(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

def kaiming_init(
    module, a=0, mode="fan_out", nonlinearity="relu", bias=0, distribution="normal"
):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class SepHead(nn.Module):
    def __init__(
            self,
            in_channels,
            heads,
            head_conv=64,
            name="",
            final_kernel=1,
            bn=False,
            init_bias=-2.19,
            directional_classifier=False,
            **kwargs,
    ):
        super(SepHead, self).__init__(**kwargs)

        self.heads = heads
        for head in self.heads:
            classes, num_conv = self.heads[head]

            fc = Sequential()
            for i in range(num_conv - 1):
                fc.add(nn.Conv2d(in_channels, head_conv,
                                 kernel_size=final_kernel, stride=1,
                                 padding=final_kernel // 2, bias=True))
                if bn:
                    fc.add(nn.BatchNorm2d(head_conv))
                fc.add(nn.ReLU(inplace=True))

            fc.add(nn.Conv2d(head_conv, classes,
                             kernel_size=final_kernel, stride=1,
                             padding=final_kernel // 2, bias=True))

            if 'hm' in head:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)

            self.__setattr__(head, fc)

        assert directional_classifier is False, "Doesn't work well with nuScenes in my experiments, please open a pull request if you are able to get it work. We really appreciate contribution for this."

    def forward(self, x):
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict

