import torch
from torch.nn import BCEWithLogitsLoss, MSELoss, L1Loss
from Utils.config import readCfg
from Constants.variableNames import AVERAGE_ALONG_AXIS, WEIGHT
import numpy as np

from Utils.plot3d import ImageObject, StructureOverlayObject, plot3d


class PytorchLoss:
    def __init__(self, cfg):
        raise NotImplementedError

    def forward(self, input, output, target, weights=None):
        raise NotImplementedError

    def getNames(self):
        return self.__class__.__name__


class BinaryCrossEntropyLoss(PytorchLoss):
    def __init__(self, cfg):
        self.classWeight = torch.tensor(readCfg(cfg, WEIGHT, 1.0, float))
        self.lossFuction = BCEWithLogitsLoss(pos_weight=self.classWeight)

    def forward(self, input, output, target, weights=None):
        return self.lossFuction(output, target)


class MeanSquareErrorLoss(PytorchLoss):
    def __init__(self, cfg):
        self.lossFuction = MSELoss(reduction="none")
        self.averageTargetAlongAxis = readCfg(cfg, AVERAGE_ALONG_AXIS, None, int)

    def forward(self, input, output, target, weights=None):
        if self.averageTargetAlongAxis is not None:
            target = torch.mean(target, dim=self.averageTargetAlongAxis)
        loss = self.lossFuction(output, target)
        if weights is not None:
            if loss[weights > 0].nelement() > 0:
                loss = (weights * loss)[weights > 0]
            else:
                loss = weights * loss  # loss will be zero anyways
        return torch.mean(loss)


class MeanAbsoluteErrorLoss(PytorchLoss):
    def __init__(self, cfg):
        self.lossFuction = L1Loss(reduction="none")
        self.averageTargetAlongAxis = readCfg(cfg, AVERAGE_ALONG_AXIS, None, int)

    def forward(self, input, output, target, weights=None):
        if self.averageTargetAlongAxis is not None:
            target = torch.mean(target, dim=self.averageTargetAlongAxis)
        loss = self.lossFuction(output, target)
        if weights is not None:
            if loss[weights > 0].nelement() > 0:
                loss = (weights * loss)[weights > 0]
            else:
                loss = weights * loss  # loss will be zero anyways
        return torch.mean(loss)


class NormalDistributionLoss(PytorchLoss):
    def __init__(self, cfg):
        return

    def forward(self, input, output, target, weights=None):
        dim = output.shape[1]
        mean = output[:, : dim // 2]  # batchSize x Channels x H x W x D
        stdev = torch.clamp(output[:, dim // 2 :], min=1e-8)
        dist = torch.distributions.normal.Normal(mean, stdev)
        return -dist.log_prob(target).mean()
