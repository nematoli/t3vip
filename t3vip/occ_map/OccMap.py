import math
from torch import nn
from torch.autograd import Function
import torch
import OccMap_cuda

torch.manual_seed(42)


# FWD/BWD pass function
class OccMapFunction(Function):
    @staticmethod
    def forward(ctx, points, fy, fx, cy, cx):
        ctx.fy, ctx.fx, ctx.cy, ctx.cx = fy, fx, cy, cx

        # Check dimensions
        batch_size, num_channels, data_height, data_width = points.size()
        assert num_channels == 3

        # Create output & temp data (3D)
        output = points.new().resize_as_(points).fill_(0)
        temp = output.narrow(1, 2, 1).fill_(float("inf"))

        indexmap = points.new()
        indexmap.resize_(batch_size, 1, data_height, data_width).fill_(-1)  # 1-channel

        OccMap_cuda.forward(
            points.contiguous(),
            indexmap.contiguous(),
            output.contiguous(),
            ctx.fy,
            ctx.fx,
            ctx.cy,
            ctx.cx,
        )
        ctx.save_for_backward(points, indexmap)  # Save for BWD pass

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Get saved tensors
        points, indexmap = ctx.saved_tensors
        assert grad_output.is_same_size(points)

        # Initialize grad input
        grad_points = points.new().resize_as_(points).fill_(0)
        OccMap_cuda.backward(
            points.contiguous(),
            indexmap.contiguous(),
            grad_points,
            grad_output.contiguous(),
            ctx.fy,
            ctx.fx,
            ctx.cy,
            ctx.cx,
        )
        # Return
        return grad_points, None, None, None, None


class OccMap(nn.Module):
    def __init__(self, intrinsics):
        super(OccMap, self).__init__()
        self.fy, self.fx, self.cy, self.cx = intrinsics["fy"], intrinsics["fx"], intrinsics["cy"], intrinsics["cx"]

        self.occ_map = OccMapFunction.apply

    def forward(self, points):
        occ_map = self.occ_map(points, self.fy, self.fx, self.cy, self.cx).narrow(1, 2, 1)
        occ_map = torch.eq(occ_map, torch.zeros_like(occ_map)).float()
        return occ_map
