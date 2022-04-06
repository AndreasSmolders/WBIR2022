import numpy as np
import torch
from torch import nn
from Utils.config import readCfg
from Model.unet3d.model import UNet3D

INPUT_CHANNELS = "inputChannels"
OUTPUT_CHANNELS = "outputChannels"
FEATURE_MAPS = "featureMaps"
LAYER_ORDER = "layerOrder"
NUMBER_OF_GROUPS = "numberOfGroups"
NUMBER_OF_ENCODERS = "numberOfEncoders"
PADDING = "padding"
KERNEL_SIZE = "kernelSize"
POOLING_SIZE = "poolingSize"


class UNetWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.inputChannels = readCfg(cfg, INPUT_CHANNELS, 1, int)
        self.outputChannels = readCfg(cfg, OUTPUT_CHANNELS, 1, int)
        self.featureMaps = readCfg(cfg, FEATURE_MAPS, 64, int)
        self.layerOrder = readCfg(cfg, LAYER_ORDER, "gcr")
        self.numberOfGroups = readCfg(cfg, NUMBER_OF_GROUPS, 8)
        self.numberOfEncoders = readCfg(cfg, NUMBER_OF_ENCODERS, 4)
        self.padding = readCfg(cfg, PADDING, "same")
        self.kernelSize = readCfg(cfg, KERNEL_SIZE, 3)
        self.poolingSize = readCfg(cfg, POOLING_SIZE, 2)
        self.model = UNet3D(
            in_channels=self.inputChannels,
            out_channels=self.outputChannels,
            f_maps=self.featureMaps,
            layer_order=self.layerOrder,
            num_groups=self.numberOfGroups,
            num_levels=self.numberOfEncoders,
            conv_padding=self.padding,
            conv_kernel_size=self.kernelSize,
            pool_kernel_size=self.poolingSize,
        )

    def forward(self, x):
        return self.model.forward(x)
