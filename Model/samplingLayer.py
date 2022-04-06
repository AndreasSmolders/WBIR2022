from torch._C import Value
import torch.nn as nn
import torch
from torch.distributions import Normal

from Utils.constants import EPS

PROBABLITY_MODEL_DIAGONAL = "diagonal"
PROBABLITY_MODEL_BLOCK_DIAGONAL = "blockDiagonal"


class NormalSamplingLayer(nn.Module):
    def __init__(
        self,
        blurringLayer=None,
        nbSamples=1,
    ):
        super().__init__()
        self.blurringLayer = blurringLayer
        self.nbSamples = nbSamples

    def forward(self, mean, stdev):
        if not self.training:
            return mean

        dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        sampleShape = torch.tensor(mean.shape)
        sampleShape[0] = sampleShape[0] * self.nbSamples

        if self.blurringLayer is None:
            eps = dist.rsample(sampleShape)[..., 0].to(mean.device)
            return mean + stdev * eps
        else:
            if self.blurringLayer.padding == 0:
                sampleShape[-3:] = (
                    sampleShape[-3:] + torch.tensor(self.blurringLayer.kernel_size) - 1
                )
            eps = dist.rsample(sampleShape)[..., 0].to(mean.device)
            return mean + stdev * self.blurringLayer(eps)


class NormalBlockSamplingLayer(NormalSamplingLayer):
    def forward(self, mean, choleskyL):
        if not self.training:
            return mean

        dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        eps = dist.rsample(mean.shape)[..., 0].to(choleskyL.device)
        dim = mean.shape[1]
        numberIndependentCovVariables = dim * (dim + 1) / 2
        assert (
            choleskyL.shape[1] == numberIndependentCovVariables
        ), "Block covariance matrix should be formable from model output shape"
        output = torch.zeros_like(mean)
        for i in range(dim):
            n = i + 1
            startIndex = int((n - 1) * (n) / 2)
            endIndex = startIndex + n
            output[:, i] += torch.sum(eps[:, :n] * choleskyL[:, startIndex:endIndex], 1)
        if self.blurringLayer is None:
            return mean + output
        else:
            raise ValueError  # wrong blurring
            return mean + self.blurringLayer(output)
