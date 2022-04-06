from ast import literal_eval
from typing import Tuple

from config import readCfg
from conversion import toIntMaybe

import torch
import torch.nn as nn
from torch.nn.functional import conv1d, conv2d, conv3d
import torch.nn.functional as F

from Utils.timing import getTime, printTiming


SAMPLE_GAUSSIAN_BLUR_KERNEL_SIZE = "sampleGaussianBlurKernelSize"
SAMPLE_GAUSSIAN_BLUR_SIGMA = "sampleGaussianBlurSigma"
SAMPLE_GAUSSIAN_BLUR_PADDING = "sampleGaussianBlurPadding"
RESCALE_CORR_MATRIX = "rescaleCorrelationMatrix"


def gaussian(window_size, sigma, offset):
    def gauss_fcn(x):
        return -((x - window_size // 2 - offset) ** 2) / float(2 * sigma ** 2)

    gauss = torch.stack(
        [torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)]
    )
    return gauss / gauss.sum()


def get_gaussian_kernel(ksize: int, sigma: float, offset: float) -> torch.Tensor:
    r"""Function that returns Gaussian filter coefficients.

    Args:
        ksize (int): filter size. It should be odd and positive.
        sigma (float): gaussian standard deviation.

    Returns:
        Tensor: 1D tensor with gaussian filter coefficients.

    Shape:
        - Output: :math:`(ksize,)`

    Examples::
    """
    if ksize != int(ksize) or ksize % 2 == 0 or ksize <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}".format(ksize))
    window_1d: torch.Tensor = gaussian(int(ksize), sigma, offset)
    return window_1d


def get_gaussian_kernelNd(
    ksize: Tuple[int], sigma: Tuple[float], offsets=None
) -> torch.Tensor:
    n = len(ksize)
    if not isinstance(ksize, tuple):
        raise TypeError("ksize must be a tuple. Got {}".format(ksize))
    if not isinstance(sigma, tuple):
        raise TypeError("sigma must be a tuple. Got {}".format(sigma))
    if offsets is None:
        offsets = [0.0] * len(ksize)
    kernels = []
    for ksizeAx, sigmaAx, offset in zip(ksize, sigma, offsets):
        kernel: torch.Tensor = get_gaussian_kernel(ksizeAx, sigmaAx, offset)
        kernels.append(kernel)

    if n == 1:
        return kernels[0]
    elif n == 2:
        return torch.matmul(kernels[0].unsqueeze(-1), kernels[1].unsqueeze(-1).t())
    elif n == 3:
        return torch.matmul(
            kernels[1].unsqueeze(-1), kernels[2].unsqueeze(-1).t()
        ) * kernels[0].unsqueeze(-1).unsqueeze(-1)
    else:
        raise NotImplementedError


def getVarianceWeights(kernelSize, smoothingSigma, shape, device, padding):
    smoothingKernel = get_gaussian_kernelNd(kernelSize, smoothingSigma)
    squaredSmoothingKernel = (
        (smoothingKernel * smoothingKernel).unsqueeze(0).unsqueeze(0)
    ).to(device)

    if padding == "same":
        onesTensor = torch.ones(shape).to(device)
        while len(onesTensor.shape) < 5:
            onesTensor = onesTensor.unsqueeze(0)
        weights = F.conv3d(onesTensor, squaredSmoothingKernel, padding=padding)
        while len(weights.shape) > len(shape):
            weights = weights[0]
        return weights
    else:
        return torch.sum(squaredSmoothingKernel)


class GaussianBlur(nn.Module):
    def __init__(
        self,
        kernel_size: Tuple[int],
        sigma: Tuple[float],
        rescale=False,
        padding="same",
    ) -> None:
        super(GaussianBlur, self).__init__()
        self.kernel_size: Tuple[int] = kernel_size
        self.sigma: Tuple[float] = sigma
        self.N = len(self.kernel_size)
        self.kernel: torch.Tensor = self.create_gaussian_kernel(kernel_size, sigma)
        self.nbChannels = 1  # will be altered on first iteration
        self.rescale = rescale
        self.rescaleTensor = None
        self.padding = padding

    @staticmethod
    def create_gaussian_kernel(kernel_size, sigma) -> torch.Tensor:
        """Returns a ND Gaussian kernel array."""
        kernel: torch.Tensor = get_gaussian_kernelNd(kernel_size, sigma)
        return kernel.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor):
        if not torch.is_tensor(x):
            raise TypeError(
                "Input x type is not a torch.Tensor. Got {}".format(type(x))
            )
        if not len(x.shape) == 2 + len(self.kernel_size):
            raise ValueError(
                "Invalid input shape, we expect BxCxHxW(xD). Got: {}".format(x.shape)
            )
        # prepare kernel in first forward pass
        c = x.shape[1]
        if self.kernel.device != x.device:
            self.kernel = self.kernel.to(x.device).to(x.dtype)
        if c != self.nbChannels:
            self.nbChannels = c
            repeats = (
                self.nbChannels,
                1,
            ) + (1,) * self.N
            self.kernel = self.kernel.repeat(repeats)

        # convolve tensor with gaussian kernel
        if self.N == 1:
            x = conv1d(
                x, self.kernel, padding=self.padding, stride=1, groups=self.nbChannels
            )
        elif self.N == 2:
            x = conv2d(
                x, self.kernel, padding=self.padding, stride=1, groups=self.nbChannels
            )
        elif self.N == 3:
            x = conv3d(
                x, self.kernel, padding=self.padding, stride=1, groups=self.nbChannels
            )
        else:
            raise NotImplementedError

        if self.rescale:
            if self.rescaleTensor is None:
                self.rescaleTensor = (
                    torch.sqrt(
                        getVarianceWeights(
                            self.kernel_size,
                            self.sigma,
                            x.shape[2:],
                            x.device,
                            self.padding,
                        )
                    )
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
            x = self.rescaleTensor * x

        return x


def createGaussianBlurLayer(cfg):
    sampleGaussianBlurKernelSize = readCfg(
        cfg, SAMPLE_GAUSSIAN_BLUR_KERNEL_SIZE, (9, 9, 9), literal_eval
    )
    sampleGaussianBlurSigma = readCfg(
        cfg, SAMPLE_GAUSSIAN_BLUR_SIGMA, (4, 4, 4), literal_eval
    )
    rescaleCorrelationMatrix = readCfg(cfg, RESCALE_CORR_MATRIX, False, literal_eval)
    gaussianBlurPadding = readCfg(cfg, SAMPLE_GAUSSIAN_BLUR_PADDING, "same", toIntMaybe)
    return GaussianBlur(
        sampleGaussianBlurKernelSize,
        sampleGaussianBlurSigma,
        rescaleCorrelationMatrix,
        gaussianBlurPadding,
    )
