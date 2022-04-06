from DL.Constants.variableNames import (
    PATCH_SIZE,
    PADDING,
)
from Utils.normalization import normalizeCT
from ast import literal_eval
from Utils.config import readCfg
from Utils.augmentations import (
    randomCropBbox,
    centerCropBbox,
    cropBbox,
    padImage,
    weightedCenterCropBbox,
)
from torch.utils.data import Dataset
import torch
from Utils.ioUtils import listDirAbsolute
from Utils.scan import readPixelArray
from Constants.variableNames import *
from Data.utils import getLabelPath
import os
from Utils.augmentations import flip, rotate90


class ImageDataset(Dataset):
    def __init__(self, cfg):
        self.idxToFiles = self.getIdxToFiles(cfg)
        self.augment = readCfg(cfg, AUGMENT, False, literal_eval)
        self.patchSize = readCfg(cfg, PATCH_SIZE, (256, 256, 96), literal_eval)
        self.padding = readCfg(cfg, PADDING, True, literal_eval)
        self.augmentFlip = readCfg(cfg, AUGMENT_FLIP, 0, float)
        self.augmentRot90 = readCfg(cfg, AUGMENT_ROTATION_90, 0, float)
        return

    def __len__(self):
        return len(self.idxToFiles)

    def __getitem__(self, idx):
        image, label = self.getImageAndLabel(idx)
        image = self.normalize(image)
        image, label, _ = self.crop(image, label)
        if self.augment:
            image, label = self.augmentImage(image, label)
        image, label = self.convertToTrainingType(image, label)
        return {INPUT: image, LABEL: label}

    def getInstancePath(self, idx):
        return self.idxToFiles[idx]

    def getIdxToFiles(self, cfg):
        patientList = readCfg(cfg, INSTANCE_LIST, [], literal_eval)
        idxToFiles = []
        preprocessedDir = cfg[TARGET_FOLDER]
        for patient in patientList:
            images = [
                path
                for path in listDirAbsolute(os.path.join(preprocessedDir, patient))
                if IMAGE in os.path.split(path)[1]
            ]
            idxToFiles += images
        return idxToFiles

    def getImageAndLabel(self, idx):
        imagePath = self.idxToFiles[idx]
        image = readPixelArray(imagePath)
        label = readPixelArray(getLabelPath(imagePath))
        return image, label

    def augmentImage(self, image, label):
        if self.augmentFlip > 0:
            image, label = self.flip(image, label)
        if self.augmentRot90 > 0:
            image, label = self.rotate90(image, label)
        return image, label

    def normalize(self, image):
        return normalizeCT(image)

    def convertToTrainingType(self, image, label):
        return (
            torch.from_numpy(image.copy()).float(),
            torch.from_numpy(label.copy()).float(),
        )

    def crop(self, image, label,weights=None):
        if self.augment:
            bbox = randomCropBbox(image.shape, self.patchSize)
        else:
            if weights is None:
                bbox = centerCropBbox(image.shape, self.patchSize)
            else:
                bbox = weightedCenterCropBbox(image.shape,self.patchSize,weights)
        image, label = cropBbox(image, bbox), cropBbox(label, bbox)
        if self.padding:
            image, label = padImage(image, self.patchSize), padImage(
                label, self.patchSize
            )
        return image, label, bbox

    def flip(self, image, label):
        return flip(image, label, self.augmentFlip)

    def rotate90(self, image, label):
        return rotate90(image, label, self.augmentRot90)
