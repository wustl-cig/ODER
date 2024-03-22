import torch
import torch.nn.functional as F
import numpy as np

if torch.__version__>'1.2.0':
    affine_grid = lambda theta, size: F.affine_grid(theta, size, align_corners=True)
    grid_sample = lambda input, grid, mode='bilinear': F.grid_sample(input, grid, align_corners=True, mode=mode)
    # print('hell0')
else:
    affine_grid = F.affine_grid
    grid_sample = F.grid_sample


# I changed into this
PI = torch.tensor(np.float32(np.pi))
SQRT2 = torch.tensor(np.sqrt(np.float32(2)))

# original code
# PI = 4*torch.ones(1).atan()
# SQRT2 = (2*torch.ones(1)).sqrt()


def fftfreq(n):
    val = 1.0/n
    results = torch.zeros(n)
    N = (n-1)//2 + 1
    p1 = torch.arange(0, N)
    results[:N] = p1
    p2 = torch.arange(-(n//2), 0)
    results[N:] = p2
    return results*val


def deg2rad(x):
    return x*PI/180


def rfft(tensor, axis=-1):
    ndim = tensor.ndim
    if axis < 0:
        axis %= ndim
    tensor = tensor.transpose(axis, ndim-1)
    fft_tensor = torch.rfft(
        tensor,
        1,
        normalized=False,
        onesided=False,
    )
    return fft_tensor.transpose(axis, ndim-1)


def irfft(tensor, axis):
    assert 0 <= axis < tensor.ndim
    tensor = tensor.transpose(axis, tensor.ndim-2)
    ifft_tensor = torch.ifft(
        tensor,
        1,
        normalized=False,
    )[..., 0]
    return ifft_tensor.transpose(axis, tensor.ndim-2)
