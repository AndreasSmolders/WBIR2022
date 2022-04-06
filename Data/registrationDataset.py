from plot3d import ImageObject, StructureOverlayObject, plot3d
import torch
from DL.Constants.variableNames import (
    DOSE,
    APPLY_WARP,
    WARPING_METHOD,
    ADD_WARP_TO_INPUT,
    LABEL_SUFFIX,
)
from DL.Data.Dataset.patientDataset import ImageDataset
from Utils.normalization import normalizeCT
from ast import literal_eval
from Utils.config import readCfg
from Utils.constants import (
    DICOM_FILE_EXTENSION,
    GZ_FILE_EXTENSION,
    NUMPY_ARRAY_EXTENSION,
    PKL_FILE_EXTENSION,
)
from Utils.augmentations import (
    cropBbox,
    padImage,
    rotate90Vector,
)
from Utils.dicoms import getPixelArray, getVoxelSpacing
from Utils.ioUtils import isEmptyFolder, listDirAbsolute, readDicom, readNumpyGz
from Utils.scan import Scan, readPixelArray
from Constants.variableNames import *
import os
import numpy as np
from Utils.augmentations import flip, flipVector, rotate90
from Utils.vectorField import VectorField
from Utils.warp import warpScan


class DVFRegistrationDataset(ImageDataset):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.applyPreWarping = readCfg(cfg, APPLY_WARP, False, literal_eval)
        self.referenceRegistrationType = readCfg(cfg, REFERENCE_REGISTRATION_TYPE, None)
        self.addDVFInput = readCfg(cfg, ADD_DVF_INPUT, True, literal_eval)

    def __getitem__(self, idx):
        image, label = self.getImageAndLabel(idx)
        weights = self.getFullBodyMask(idx)
        image, label, bbox = self.crop(image, label,weights)
        weights = self.getBodyMask(idx,bbox)
        dvf = self.getDVF(idx, bbox)
        if self.augment:
            image, label, dvf = self.augmentImage(image, label, dvf)
        if dvf is not None:
            image = np.concatenate((image, dvf), 0)
        image, label = self.convertToTrainingType(image, label)
        return {INPUT: image, LABEL: label, WEIGHTS: weights}

    def augmentImage(self, image, label, dvf):
        if self.augmentFlip > 0:
            image, label, dvf = self.flip(image, label, dvf)
        if self.augmentRot90 > 0:
            image, label, dvf = self.rotate90(image, label, dvf)
        return image, label, dvf

    def flip(self, image, label, dvf):
        return flipVector(image, label, dvf, self.augmentFlip)

    def rotate90(self, image, label, dvf):
        return rotate90Vector(image, label, dvf, self.augmentRot90)

    def getIdxToFiles(self, cfg):
        patientList = readCfg(cfg, INSTANCE_LIST, [], literal_eval)
        idxToFiles = []
        preprocessedDir = cfg[TARGET_FOLDER]
        for patient in patientList:
            images = [
                path
                for path in listDirAbsolute(os.path.join(preprocessedDir, patient))
                if not isEmptyFolder(path)
            ]
            idxToFiles += images
        return idxToFiles

    def getImageAndLabel(self, idx):
        imageFolder = self.idxToFiles[idx]
        image, voxelSpacing = self.getImages(imageFolder)
        label = self.getLabels(imageFolder, voxelSpacing)
        return image, label

    def getImages(self, imageFolder):
        imageNames = os.path.split(imageFolder)[1].split("_")
        assert len(imageNames) == 2
        fixed = Scan(self.getImageName(imageFolder, imageNames[0])).getPixelArray()
        moving = Scan(self.getImageName(imageFolder, imageNames[1]))
        if self.applyPreWarping:
            dvf = VectorField(self.getDVFName(imageFolder))
            moving = warpScan(moving, dvf.getPixelArray())
        voxelSpacing = moving.getVoxelSpacing()
        moving = moving.getPixelArray()
        image = normalizeCT(np.concatenate((fixed, moving), 0))
        return image, voxelSpacing

    def getDVF(self, idx, bbox):
        if (
            self.referenceRegistrationType
            and not self.applyPreWarping
            and self.addDVFInput
        ):
            dvf = VectorField(self.getDVFName(self.idxToFiles[idx])).getPixelArray()
            dvf = cropBbox(dvf, bbox)
            if self.padding:
                dvf = padImage(dvf, self.patchSize)
            return dvf
        else:
            return None

    def getDVFName(self, targetFolderName):
        return os.path.join(
            targetFolderName,
            f"{LABEL}_{self.referenceRegistrationType}.{PKL_FILE_EXTENSION}",
        )

    def getLabels(self, imageFolder, voxelSpacing):
        return readNumpyGz(
            self.getLabelName(imageFolder, self.referenceRegistrationType)
        )

    def getLabelName(self, targetFolderName, referenceRegistrationType):
        if referenceRegistrationType is None:
            referenceRegistrationType = ""
        return os.path.join(
            targetFolderName,
            "{}_{}_{}.{}.{}".format(
                LABEL,
                STANDARD_DEVIATION_ABBREVIATION,
                referenceRegistrationType,
                NUMPY_ARRAY_EXTENSION,
                GZ_FILE_EXTENSION,
            ),
        )

    def getImageName(self, targetFolderName, scanName):
        return self.getName(targetFolderName, scanName, IMAGE, PKL_FILE_EXTENSION)

    def getName(self, targetFolderName, scanName, type, extension=DICOM_FILE_EXTENSION):
        filePath = os.path.join(
            targetFolderName,
            "{}_{}.{}".format(type, scanName, extension),
        )
        if os.path.isfile(filePath):
            return filePath
        elif os.path.isfile(filePath + ".gz"):
            return filePath + ".gz"
        else:
            raise ValueError(f"File {filePath} not found")

    def getFullBodyMask(self,idx):
        return readPixelArray(self.getBodyMaskName(self.idxToFiles[idx]))[0]

    def getBodyMask(self, idx, cropBox):
        weights = readPixelArray(self.getBodyMaskName(self.idxToFiles[idx]))
        weights = cropBbox(weights, cropBox)
        if self.padding:
            weights = padImage(weights, self.patchSize)
        weights = torch.from_numpy(np.repeat(weights.copy(), 3, 0))
        return weights

    def getBodyMaskName(self, targetFolderName):
        return os.path.join(
            targetFolderName,
            "{}.{}.{}".format(BODY_MASK, PKL_FILE_EXTENSION, GZ_FILE_EXTENSION),
        )


class DoseRegistrationDataset(DVFRegistrationDataset):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.addDoseGradient = readCfg(cfg, ADD_DOSE_GRADIENT, False, literal_eval)

    def getDose(self, imageFolder):
        imageNames = os.path.split(imageFolder)[1].split("_")
        assert len(imageNames) == 2
        movingDose = Scan(self.getDoseName(imageFolder, imageNames[1]))
        if self.applyPreWarping:
            dvf = VectorField(self.getDVFName(imageFolder))
            movingDose = warpScan(movingDose, dvf.getPixelArray())
        movingDose = movingDose.getPixelArray()

        if self.addDoseGradient:
            grad = np.gradient(movingDose[0])
            movingDose = np.concatenate((movingDose, grad), 0)

        return movingDose

    def getImageAndLabel(self, idx):
        imageFolder = self.idxToFiles[idx]
        image, voxelSpacing = self.getImages(imageFolder)
        dose = self.getDose(imageFolder)
        image = np.concatenate((image, dose), 0)
        label = self.getLabels(imageFolder, voxelSpacing)
        return image, label

    def getDoseName(self, targetFolderName, scanName):
        return self.getName(targetFolderName, scanName, DOSE, PKL_FILE_EXTENSION)

    def getLabelName(self, targetFolderName, referenceRegistrationType):
        if referenceRegistrationType is None:
            referenceRegistrationType = ""
        return os.path.join(
            targetFolderName,
            f"{LABEL}_{DOSE}_{STANDARD_DEVIATION_ABBREVIATION}_{referenceRegistrationType}.{NUMPY_ARRAY_EXTENSION}.{GZ_FILE_EXTENSION}".format(),
        )

    def getBodyMask(self, idx, cropBox):
        weights = readPixelArray(self.getBodyMaskName(self.idxToFiles[idx]))
        weights = cropBbox(weights, cropBox)
        if self.padding:
            weights = padImage(weights, self.patchSize)
        return torch.from_numpy(weights)


class UnsupervisedRegistrationDataset(DVFRegistrationDataset):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.applyPreWarping = readCfg(cfg, APPLY_WARP, False, literal_eval)
        self.addDVFInput = readCfg(cfg, ADD_WARP_TO_INPUT, False, literal_eval)
        self.referenceRegistrationType = readCfg(
            cfg, WARPING_METHOD, "bSplineLena", None
        )

