import torch
import torch.nn as nn
from torch.autograd import Function, Variable

from . import pointnet2_stack_cuda as pointnet2
from .pointnet2_utils import grouping_operation


class BallQueryDeform(Function):

    @staticmethod
    def forward(ctx, nsample: int, xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor,
                new_xyz: torch.Tensor, new_xyz_r: torch.Tensor, new_xyz_batch_cnt: torch.Tensor):
        """
        :param ctx:
        :param nsample: int, maximum number of features in the balls
        :param xyz: (N1 + N2 ..., 3) xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3) centers of the ball query
        :param new_xyz_r: (M1 + M2 ..., 1) radius for each new point
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :return:
            idx: (M1 + M2, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert new_xyz_r.is_contiguous()
        assert new_xyz_batch_cnt.is_contiguous()
        assert xyz.is_contiguous()
        assert xyz_batch_cnt.is_contiguous()
        assert int(xyz.size(0)) == int(sum(xyz_batch_cnt))
        assert int(new_xyz.size(0)) == int(sum(new_xyz_batch_cnt))
        assert int(new_xyz_r.size(0)) == int(sum(new_xyz_batch_cnt))

        B = xyz_batch_cnt.shape[0]
        M = new_xyz.shape[0]
        idx = torch.cuda.IntTensor(M, nsample).zero_()

        pointnet2.ball_query_deform_wrapper(B, M, nsample, new_xyz, new_xyz_r, new_xyz_batch_cnt, xyz, xyz_batch_cnt, idx)
        empty_ball_mask = (idx[:, 0] == -1)
        idx[empty_ball_mask] = 0
        return idx, empty_ball_mask

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None

ball_query_deform = BallQueryDeform.apply

class QueryAndGroupPyramid(nn.Module):
    def __init__(self, nsample, use_xyz = True):
        super().__init__()
        self.nsample = nsample
        self.use_xyz = use_xyz

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_r, new_xyz_batch_cnt, features):
        assert xyz.shape[0] == xyz_batch_cnt.sum(), 'xyz: %s, xyz_batch_cnt: %s' % (str(xyz.shape), str(new_xyz_batch_cnt))
        assert new_xyz.shape[0] == new_xyz_batch_cnt.sum(), \
            'new_xyz: %s, new_xyz_batch_cnt: %s' % (str(new_xyz.shape), str(new_xyz_batch_cnt))

        # add no_grad
        with torch.no_grad():
            idx, empty_ball_mask = ball_query_deform(self.nsample, xyz, xyz_batch_cnt, new_xyz, new_xyz_r, new_xyz_batch_cnt)
        grouped_xyz = grouping_operation(xyz, xyz_batch_cnt, idx, new_xyz_batch_cnt)  # (M1 + M2, 3, nsample)
        grouped_xyz -= new_xyz.unsqueeze(-1)

        grouped_xyz[empty_ball_mask] = 0

        if features is not None:
            grouped_features = grouping_operation(features, xyz_batch_cnt, idx, new_xyz_batch_cnt)  # (M1 + M2, C, nsample)
            grouped_features[empty_ball_mask] = 0
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (M1 + M2 ..., C + 3, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features, idx

class QueryAndGroupPyramidAttention(nn.Module):
    def __init__(self, nsample):
        super().__init__()
        self.nsample = nsample

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_r, new_xyz_batch_cnt, features):
        assert xyz.shape[0] == xyz_batch_cnt.sum(), 'xyz: %s, xyz_batch_cnt: %s' % (str(xyz.shape), str(new_xyz_batch_cnt))
        assert new_xyz.shape[0] == new_xyz_batch_cnt.sum(), \
            'new_xyz: %s, new_xyz_batch_cnt: %s' % (str(new_xyz.shape), str(new_xyz_batch_cnt))

        # add no_grad
        with torch.no_grad():
            idx, empty_ball_mask = ball_query_deform(self.nsample, xyz, xyz_batch_cnt, new_xyz, new_xyz_r, new_xyz_batch_cnt)
        grouped_xyz = grouping_operation(xyz, xyz_batch_cnt, idx, new_xyz_batch_cnt)  # (M1 + M2, 3, nsample)
        grouped_xyz -= new_xyz.unsqueeze(-1)
        grouped_xyz[empty_ball_mask] = 0

        assert features is not None
        grouped_features = grouping_operation(features, xyz_batch_cnt, idx, new_xyz_batch_cnt)  # (M1 + M2, C, nsample)
        grouped_features[empty_ball_mask] = 0
        grouped_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (M1 + M2 ..., C + 3, nsample)

        return grouped_xyz.contiguous(), grouped_features.contiguous(), empty_ball_mask

class QueryAndGroupDeform(nn.Module):
    def __init__(self, temperature: float, nsample: int, use_xyz: bool = True):
        """
        :param temperature: float, sigmoid temperature
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.temperature, self.nsample, self.use_xyz = temperature, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor,
                new_xyz: torch.Tensor, new_xyz_r: torch.Tensor, new_xyz_batch_cnt: torch.Tensor,
                features: torch.Tensor = None, temperature_decay: float = None):
        """
        :param xyz: (N1 + N2 ..., 3) xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3) centers of the ball query
        :param new_xyz_r: (M1 + M2 ..., 1) radius for each new point
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of features to group
        :return:
            new_features: (M1 + M2, C, nsample) tensor
        """
        assert xyz.shape[0] == xyz_batch_cnt.sum(), 'xyz: %s, xyz_batch_cnt: %s' % (str(xyz.shape), str(new_xyz_batch_cnt))
        assert new_xyz.shape[0] == new_xyz_batch_cnt.sum(), \
            'new_xyz: %s, new_xyz_batch_cnt: %s' % (str(new_xyz.shape), str(new_xyz_batch_cnt))

        real_temperature = self.temperature * temperature_decay

        # add explore window for each radius, window size depends on temperature
        # temperature coefficient 5 is for clippping points with weights < 0.01
        explore_new_xyz_r = new_xyz_r + real_temperature * 5

        # idx: (M1 + M2 ..., nsample), empty_ball_mask: (M1 + M2 ...)
        # add no_grad
        with torch.no_grad():
            idx, empty_ball_mask = ball_query_deform(self.nsample, xyz, xyz_batch_cnt, new_xyz, explore_new_xyz_r, new_xyz_batch_cnt)
        grouped_xyz = grouping_operation(xyz, xyz_batch_cnt, idx, new_xyz_batch_cnt)  # (M1 + M2, 3, nsample)
        grouped_xyz -= new_xyz.unsqueeze(-1)

        grouped_xyz[empty_ball_mask] = 0

        dist_xyz = torch.sqrt((grouped_xyz * grouped_xyz).sum(dim = 1, keepdim = False)) # (M1 + M2, nsample)

        weights = 1 - torch.sigmoid((dist_xyz - new_xyz_r) / real_temperature) # (M0 + M2, nsample) - (M1 + M2, 1) -> (M1 + M2, nsample)

        if features is not None:
            grouped_features = grouping_operation(features, xyz_batch_cnt, idx, new_xyz_batch_cnt)  # (M1 + M2, C, nsample)
            grouped_features[empty_ball_mask] = 0
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (M1 + M2 ..., C + 3, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        # weights * feature
        # new_features = weights.unsqueeze(1) * new_features

        return new_features, weights, idx


if __name__ == '__main__':
    pass
