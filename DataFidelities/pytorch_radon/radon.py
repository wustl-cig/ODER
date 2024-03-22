import torch
from torch import nn
import torch.nn.functional as F
# import time
from .utils import PI, SQRT2, deg2rad, affine_grid, grid_sample
from .filters import RampFilter

class Radon_grad(nn.Module):
    def __init__(self, in_size):
        super(Radon_grad, self).__init__()
        self.in_size = in_size

    def forward(self, x, all_grids, circle=False):
        N, C, W, H = x.shape
        assert (W == H)

        if not circle:
            diagonal = SQRT2 * W
            pad = int((diagonal - W).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            x = F.pad(x, (pad_width[0], pad_width[1], pad_width[0], pad_width[1]))

        N, C, W, _ = x.shape

        L, _, _, _ = all_grids.shape
        grid = all_grids.view(L * W, W, 2).expand(N, -1, -1, -1)
        x_sampled = F.grid_sample(x, grid, align_corners=True)
        out = x_sampled.view(N, C, L, W, W).sum(dim=3).transpose(-1, -2)
        return out

class IRadon_grad(nn.Module):
    def __init__(self, in_size, use_filter=None):
        super(IRadon_grad, self).__init__()
        self.in_size = in_size
        self.filter = use_filter if use_filter is not None else lambda x: x

    def forward(self, x, all_grids, theta, circle=False):
        it_size = x.shape[2]
        ch_size = x.shape[1]

        x = self.filter(x).to(x.device)

        N, C, W, _ = x.shape
        L, _, _, _ = all_grids.shape
        grid = all_grids.view(L * W, W, 2).expand(N, -1, -1, -1)
        x_sampled = F.grid_sample(x, grid, align_corners=True)
        reco = x_sampled.view(N, C, L, W, W).sum(dim=2)

        if not circle:
            W = self.in_size
            diagonal = it_size
            pad = int(torch.tensor(diagonal - W, dtype=torch.float).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            reco = F.pad(reco, (-pad_width[0], -pad_width[1], -pad_width[0], -pad_width[1])).to(x.device)

        reco = reco * PI.item() / (2 * len(theta))

        return reco
    @staticmethod
    def _create_grids(angles, xgrid, ygrid):
        grid_size = xgrid.shape[0]

        X = torch.linspace(-1.0, 1.0, len(angles), device=angles.device).unsqueeze(-1).unsqueeze(-1).expand(-1, grid_size, grid_size)
        rad = deg2rad(angles).unsqueeze(-1).unsqueeze(-1)
        c, s = rad.cos(), rad.sin()
        Y = xgrid.unsqueeze(0) * c - ygrid * s
        return torch.stack((X, Y), dim=-1)

class Radon(nn.Module):
    def __init__(self, in_size=None, theta=None, circle=True, dtype=torch.float, device=torch.device('cuda')):
        super(Radon, self).__init__()
        self.circle = circle
        self.theta = theta
        if theta is None:
            self.theta = torch.arange(180)
        self.dtype = dtype
        self.all_grids = None
        self.device = device

        if in_size is not None:
            self.all_grids = self._create_grids(self.theta, in_size, circle)

    def forward(self, x):
        N, C, W, H = x.shape
        assert (W == H)

        if self.all_grids is None:
            self.all_grids = self._create_grids(self.theta, W, self.circle)

        if not self.circle:
            diagonal = SQRT2 * W
            pad = int((diagonal - W).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            x = F.pad(x, (pad_width[0], pad_width[1], pad_width[0], pad_width[1]))

        N, C, W, _ = x.shape

        L, _, __, _ = self.all_grids.shape
        grid = self.all_grids.to(x.device).view(L * W, W, 2).expand(N, -1, -1, -1)
        x_sampled = F.grid_sample(x, grid, align_corners=True)
        out = x_sampled.view(N, C, L, W, W).sum(dim=3).transpose(-1, -2)
        return out

    def _create_grids(self, angles, grid_size, circle):
        if not circle:
            grid_size = int((SQRT2 * grid_size).ceil())

        rad = deg2rad(angles)
        c, s = rad.cos(), rad.sin()
        R = torch.stack((torch.stack((c, s, torch.zeros_like(c)), dim=-1),
                         torch.stack((-s, c, torch.zeros_like(c)), dim=-1)), dim=-2)
        return F.affine_grid(R, (R.shape[0], 1, grid_size, grid_size), align_corners=True)


class IRadon(nn.Module):
    def __init__(self, in_size=None, theta=None, circle=True,
                 use_filter=RampFilter(), out_size=None, dtype=torch.float, device=torch.device('cuda')):
        super(IRadon, self).__init__()
        self.circle = circle
        self.theta = theta if theta is not None else torch.arange(180)
        self.out_size = out_size
        self.in_size = in_size
        self.dtype = dtype
        self.ygrid, self.xgrid, self.all_grids = None, None, None
        self.device = device

        if in_size is not None:
            self.ygrid, self.xgrid = self._create_yxgrid(in_size, circle, device=self.device)
            self.all_grids = self._create_grids(self.theta, in_size, circle, device=self.device)
        self.filter = use_filter if use_filter is not None else lambda x: x

    def forward(self, x):
        it_size = x.shape[2]
        ch_size = x.shape[1]

        x = self.filter(x).to(x.device)

        N, C, W, _ = x.shape
        L, _, __, _ = self.all_grids.shape
        grid = self.all_grids.to(x.device).view(L * W, W, 2).expand(N, -1, -1, -1)
        x_sampled = F.grid_sample(x, grid, align_corners=True)
        reco = x_sampled.view(N, C, L, W, W).sum(dim=2)

        if not self.circle:
            W = self.in_size
            diagonal = it_size
            pad = int(torch.tensor(diagonal - W, dtype=torch.float).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            reco = F.pad(reco, (-pad_width[0], -pad_width[1], -pad_width[0], -pad_width[1])).to(x.device)

        reco = reco * PI.item() / (2 * len(self.theta))

        return reco

    def _create_yxgrid(self, in_size, circle, device='cuda'):
        if not circle:
            in_size = int((SQRT2 * in_size).ceil())
        unitrange = torch.linspace(-1, 1, in_size, dtype=self.dtype).to(device)
        return torch.meshgrid(unitrange, unitrange)

    def _XYtoT(self, theta, device='cuda'):
        T = self.xgrid * (deg2rad(theta)).cos() - self.ygrid * (deg2rad(theta)).sin().to(device)
        return T

    def _create_grids(self, angles, grid_size, circle, device='cuda'):
        if not circle:
            grid_size = int((SQRT2 * grid_size).ceil())

        X = torch.linspace(-1.0, 1.0, len(angles), device=angles.device).unsqueeze(-1).unsqueeze(-1).expand(-1, grid_size, grid_size)
        rad = deg2rad(angles).unsqueeze(-1).unsqueeze(-1)
        c, s = rad.cos(), rad.sin()
        Y = self.xgrid.unsqueeze(0) * c - self.ygrid * s
        return torch.stack((X, Y), dim=-1)
