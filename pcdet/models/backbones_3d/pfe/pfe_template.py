import torch
import torch.nn as nn

from ....ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ....ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from ....ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
from ....utils import common_utils


class PFETemplate(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

    def get_sampled_points(self, batch_dict):
        batch_size = batch_dict['batch_size']
        if self.model_cfg.POINT_SOURCE == 'raw_points':
            src_points = batch_dict['points'][:, 1:4]
            src_feas = batch_dict['points'][:, 4:] if batch_dict['points'].shape[-1] > 4 else None
            batch_indices = batch_dict['points'][:, 0].long()
        elif self.model_cfg.POINT_SOURCE == 'voxel_centers':
            src_points = common_utils.get_voxel_centers(
                batch_dict['voxel_coords'][:, 1:4],
                downsample_times=1,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            src_feas = None
            batch_indices = batch_dict['voxel_coords'][:, 0].long()
        else:
            raise NotImplementedError
        keypoints_list = []
        fea_list = []
        for bs_idx in range(batch_size):
            bs_mask = (batch_indices == bs_idx)
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)
            sampled_feas = src_feas[bs_mask].unsqueeze(dim=0) if src_feas is not None else None
            if self.model_cfg.SAMPLE_METHOD == 'FPS':
                cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(
                    sampled_points[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS
                ).long()

                if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS:
                    empty_num = self.model_cfg.NUM_KEYPOINTS - sampled_points.shape[1]
                    cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]

                keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)
                feas = sampled_feas[0][cur_pt_idxs[0]].unsqueeze(dim=0) if sampled_feas is not None else None

            elif self.model_cfg.SAMPLE_METHOD == 'Proposal':
                if batch_dict.get('batch_index', None) is not None:
                    assert batch_dict['batch_box_preds'].shape.__len__() == 2
                    proposal_mask = (batch_dict['batch_index'] == bs_idx)
                else:
                    assert batch_dict['batch_box_preds'].shape.__len__() == 3
                    proposal_mask = bs_idx
                cls_preds = batch_dict['batch_cls_preds']
                if not isinstance(cls_preds, list):
                    cls_scores = torch.sigmoid(torch.max(cls_preds[proposal_mask], -1)[0])
                else:
                    cls_scores = [torch.sigmoid(torch.max(x[proposal_mask], -1)[0]) for x in cls_preds]
                    cls_scores = torch.cat(cls_scores, 0)
                box_preds = batch_dict['batch_box_preds'][proposal_mask]
                proposals = box_preds[cls_scores > self.model_cfg.get('PROPOSAL_SCORE_THRESH', 0.1)]
                if len(proposals) > 0:
                    no_proposal = False
                    point_mask = (points_in_boxes_gpu(sampled_points, proposals.unsqueeze(dim=0)) > 0)[0]
                    idx = torch.arange(sampled_points.shape[1]).to(sampled_points.device)

                    proposal_pts, proposal_idx = sampled_points[0][point_mask].unsqueeze(dim=0), idx[point_mask].unsqueeze(dim=0)
                    bg_pts, bg_idx = sampled_points[0][~point_mask].unsqueeze(dim=0), idx[~point_mask].unsqueeze(dim=0)
                    num_proposal, num_all = proposal_pts.shape[1], sampled_points.shape[1]
                else:
                    no_proposal = True
                    num_all = num_proposal = 0

                if no_proposal or num_all < self.model_cfg.NUM_KEYPOINTS:
                    cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(
                        sampled_points[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS
                    ).long()

                    if num_all < self.model_cfg.NUM_KEYPOINTS:
                        empty_num = self.model_cfg.NUM_KEYPOINTS - sampled_points.shape[1]
                        cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]
                elif num_proposal < self.model_cfg.NUM_KEYPOINTS:
                    cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(
                        bg_pts[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS - num_proposal
                    ).long()
                    cur_pt_idxs = torch.cat([proposal_idx, bg_idx[0][cur_pt_idxs[0]].unsqueeze(dim=0)], -1)
                else:
                    cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(
                        proposal_pts[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS
                    ).long()
                    cur_pt_idxs = proposal_idx[0][cur_pt_idxs[0]].unsqueeze(dim=0)

                keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)
                feas = sampled_feas[0][cur_pt_idxs[0]].unsqueeze(dim=0) if sampled_feas is not None else None

            elif self.model_cfg.SAMPLE_METHOD == 'FastFPS':
                raise NotImplementedError
            else:
                raise NotImplementedError

            keypoints_list.append(keypoints)
            fea_list.append(feas)

        keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3)
        feas = torch.cat(fea_list, 0) if fea_list[0] is not None else None
        return keypoints, feas

    def forward(self, **kwargs):
        raise NotImplementedError
