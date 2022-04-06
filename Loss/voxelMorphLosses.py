from conversion import toIntMaybe
import torch
import torch.nn.functional as F
import numpy as np
import math
from DL.Constants.variableNames import (
    FIXED_VOXEL_SPACING,
    REGULARIZATION_FACTOR,
    REGULARIZATION_NORM,
    LAMBDA_PRIOR,
    IMAGE_VARIANCE,
)
from DL.Loss.pytorchLosses import (
    MeanAbsoluteErrorLoss,
    MeanSquareErrorLoss,
    PytorchLoss,
)
from ast import literal_eval
from DL.Model.VoxelMorph import (
    PRECISION_LOSS_MODEL,
    SAMPLE_GAUSSIAN_BLUR,
    FIRST_ORDER,
    SECOND_ORDER,
)
from DL.Model.samplingLayer import (
    PROBABLITY_MODEL_BLOCK_DIAGONAL,
    PROBABLITY_MODEL_DIAGONAL,
)
from DL.Model.gaussianBlur import (
    createGaussianBlurLayer,
    get_gaussian_kernelNd,
    SAMPLE_GAUSSIAN_BLUR_PADDING,
    SAMPLE_GAUSSIAN_BLUR_SIGMA,
    SAMPLE_GAUSSIAN_BLUR_KERNEL_SIZE,
    RESCALE_CORR_MATRIX,
)
from Utils.config import readCfg
from Utils.array import getDimensions
from Utils.constants import EPS

TRACE_COEFFICIENT = "traceCoefficient"
PRECISION_COEFFICIENT = "precisionCoefficient"
LOGSIGMA_COEFFICIENT = "logSigmaCoefficient"


class NCC(PytorchLoss):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def forward(self, x, y_pred, y_true):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], (
            "volumes should be 1 to 3 dimensions. found: %d" % ndims
        )

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(y_pred.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = 1
            padding = pad_no
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, "conv%dd" % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class Dice(PytorchLoss):
    """
    N-D dice for segmentation
    """

    def forward(self, x, y_pred, y_true):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad(PytorchLoss):
    """
    N-D gradient loss.
    """

    def __init__(self, cfg):
        self.penalty = readCfg(cfg, REGULARIZATION_NORM, "l2", None)
        self.lossPreFacotr = readCfg(cfg, REGULARIZATION_FACTOR, 0.02, float)

    def forward(self, x, y_pred, _):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == "l2":
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.lossPreFacotr is not None:
            grad *= self.lossPreFacotr
        return grad


class RegularizedMSE(PytorchLoss):
    def __init__(self, cfg):
        self.loss = MeanSquareErrorLoss(cfg)
        self.regularization = Grad(cfg)

    def forward(self, input, output, target, weights=None):
        flowField = output[1]
        movedImage = output[0]
        fixedImage = input[:, 0:1]
        return self.loss.forward(
            input, movedImage, fixedImage
        ) + self.regularization.forward(input, flowField, None)


class MSEProb(PytorchLoss):
    def __init__(self, cfg):
        self.loss = MeanSquareErrorLoss(cfg)
        self.imageVariance = readCfg(cfg, IMAGE_VARIANCE, 0.02, float)

    def forward(self, input, output, target, weights=None):
        movedImage = output[0]
        fixedImage = input[:, 0:1]
        mseLoss = (
            1
            / (2 * self.imageVariance)
            * self.loss.forward(input, movedImage, fixedImage)
        )
        return mseLoss


class KLDivergence(PytorchLoss):
    """
    Kullback–Leibler divergence for probabilistic flows.
    """

    def __init__(self, cfg):
        self.lambdaPrior = readCfg(cfg, LAMBDA_PRIOR, 20, float)
        self.fixedVoxelSpacing = readCfg(
            cfg, FIXED_VOXEL_SPACING, (1.0, 1.0, 1.0), literal_eval
        )
        self.traceCoefficient = readCfg(cfg, TRACE_COEFFICIENT, 1, float)
        self.precisionCoefficient = readCfg(cfg, PRECISION_COEFFICIENT, 1, float)
        self.logCoefficient = readCfg(cfg, LOGSIGMA_COEFFICIENT, 1, float)
        self.precLossModel = readCfg(cfg, PRECISION_LOSS_MODEL, FIRST_ORDER, None)

        self.D = None

    def forward(self, inputs, output, target, weights=None):
        """
        KL loss
        """
        z = output[2]
        ndims = getDimensions(z) - 2

        precisionTerm = (
            0.5 * ndims * self.precisionCoefficient * self.precisionLoss(z, ndims)
        )
        sigmaTraceTerm = (
            0.5 * ndims * self.traceCoefficient * self.sigmaTraceLoss(z, ndims)
        )
        logSigma = -0.5 * ndims * self.logCoefficient * self.logSigmaLoss(z, ndims)

        # ndims because we averaged over dimensions as well
        return (sigmaTraceTerm, precisionTerm, logSigma)

    def sigmaTraceLoss(self, z, ndims):
        return 0

    def logSigmaLoss(self, z, ndims):
        return 0

    def precisionLoss(self, z, ndims):
        """
        a more manual implementation of the precision matrix term
                mu * P * mu    where    P = D - A
        where D is the degree matrix and A is the adjacency matrix
                mu * P * mu = 0.5 * sum_i mu_i sum_j (mu_i - mu_j) = 0.5 * sum_i,j (mu_i - mu_j) ^ 2
        where j are neighbors of i
        Note: could probably do with a difference filter,
        but the edges would be complicated unless tensorflow allowed for edge copying
        """
        meanField = z[:, :ndims]
        precLoss = 0
        if self.precLossModel == FIRST_ORDER:
            for i in range(ndims):
                d = i + 2  # batch size + vector size => +2
                # permute dimensions to put the ith dimension first
                r = [d, *range(d), *range(d + 1, ndims + 2)]
                y = torch.permute(meanField, r)
                df = (y[1:] - y[:-1]) * self.fixedVoxelSpacing[i]
                precLoss += torch.mean(df * df)  # I think we're missing a factor 2 here

            return 0.5 * self.lambdaPrior * precLoss / ndims
        elif self.precLossModel == SECOND_ORDER:
            for i in range(ndims):
                # first part d^2/dx^2,d^2/dy^2,d^2/dz^2
                d = i + 2  # batch size + vector size => +2
                # permute dimensions to put the ith dimension first
                r = [d, *range(d), *range(d + 1, ndims + 2)]
                y = torch.permute(meanField, r)
                df = (-2 * y[1:-1] + y[:-2] + y[2:]) * self.fixedVoxelSpacing[i]
                precLoss += torch.mean(df * df)

            # d^2/dxdy
            df = (
                meanField[:, :, :-2, :-2, 1:-1]
                + meanField[:, :, 2:, 2:, 1:-1]
                - meanField[:, :, :-2, 2:, 1:-1]
                - meanField[:, :, 2:, :-2, 1:-1]
            ) / 4
            precLoss += torch.mean(2 * df * df)
            # d^2/dxdz
            df = (
                meanField[:, :, :-2, 1:-1, :-2]
                + meanField[:, :, 2:, 1:-1, 2:]
                - meanField[:, :, :-2, 1:-1, 2:]
                - meanField[:, :, 2:, 1:-1, :-2]
            ) / 4
            precLoss += torch.mean(2 * df * df)
            # d^2/dzdy
            df = (
                meanField[:, :, 1:-1, :-2, :-2]
                + meanField[:, :, 1:-1, 2:, 2:]
                - meanField[:, :, 1:-1, :-2, 2:]
                - meanField[:, :, 1:-1, 2:, :-2]
            ) / 4
            precLoss += torch.mean(2 * df * df)
            return self.lambdaPrior * precLoss / ndims
        else:
            raise NotImplementedError

    def _adj_filt(self, ndims):
        """
        compute an adjacency filter that, for each feature independently,
        has a '1' in the immediate neighbor, and 0 elsewhere.
        so for each filter, the filter has 2^ndims 1s.
        the filter is then setup such that feature i outputs only to feature i
        """
        # inner filter, that is 3x3x...
        filt_inner = np.zeros([3] * ndims)
        for j in range(ndims):
            o = [[1]] * ndims
            o[j] = [0, 2]
            filt_inner[np.ix_(*o)] = 1

            # full filter, that makes sure the inner filter is applied
            # ith feature to ith feature

        filt = np.zeros([ndims, ndims] + [3] * ndims)
        for i in range(ndims):
            filt[i, i] = filt_inner

        return filt

    def getDegreeMatrix(self, shape):
        # get shape stats
        ndims = len(shape)
        sz = [ndims, *shape]

        z = torch.ones([1] + sz)
        if self.precLossModel == FIRST_ORDER:
            adjacencyFilter = torch.from_numpy(self._adj_filt(ndims)).float()
            if ndims == 3:
                return F.conv3d(z, adjacencyFilter, padding="same")
            elif ndims == 2:
                return F.conv2d(z, adjacencyFilter, padding="same")
            elif ndims == 1:
                return F.conv1d(z, adjacencyFilter, padding="same")
            else:
                raise NotImplementedError
        elif self.precLossModel == SECOND_ORDER:
            return 6.5 * ndims * z
        else:
            raise NotImplementedError

    def getNames(self):
        return ("KL trace sigma", "KL precisionterm", "KL log sigma")


class KLDivergenceProbabilisticDiagonal(KLDivergence):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.sampleGaussianBlur = readCfg(
            cfg, SAMPLE_GAUSSIAN_BLUR, False, literal_eval
        )
        if self.sampleGaussianBlur:
            self.blurringLayer = createGaussianBlurLayer(cfg)
        else:
            self.blurringLayer = None

        self.rescaleCorrelationMatrix = readCfg(
            cfg, RESCALE_CORR_MATRIX, False, literal_eval
        )
        self.K = None

    def getSquaredSmoothingKernel(self, dim, neighbour, device):
        smoothingKernel = self.blurringLayer.kernel[0, 0].to(device)
        squaredSmoothingKernel = torch.zeros_like(smoothingKernel).to(device)
        if dim == 0:
            if neighbour >= 0:
                squaredSmoothingKernel[abs(neighbour) :] = (
                    smoothingKernel[: -abs(neighbour)]
                    * smoothingKernel[abs(neighbour) :]
                )
            else:
                squaredSmoothingKernel[: -abs(neighbour)] = (
                    smoothingKernel[: -abs(neighbour)]
                    * smoothingKernel[abs(neighbour) :]
                )
        if dim == 1:
            if neighbour >= 0:
                squaredSmoothingKernel[:, abs(neighbour) :] = (
                    smoothingKernel[:, : -abs(neighbour)]
                    * smoothingKernel[:, abs(neighbour) :]
                )
            else:
                squaredSmoothingKernel[:, : -abs(neighbour)] = (
                    smoothingKernel[:, : -abs(neighbour)]
                    * smoothingKernel[:, abs(neighbour) :]
                )
        if dim == 2:
            if neighbour >= 0:
                squaredSmoothingKernel[:, :, abs(neighbour) :] = (
                    smoothingKernel[:, :, : -abs(neighbour)]
                    * smoothingKernel[:, :, abs(neighbour) :]
                )
            else:
                squaredSmoothingKernel[:, :, : -abs(neighbour)] = (
                    smoothingKernel[:, :, : -abs(neighbour)]
                    * smoothingKernel[:, :, abs(neighbour) :]
                )

        return squaredSmoothingKernel.unsqueeze(0).unsqueeze(0)

    def getCorrelationMatrix(self, shape, device):
        smoothingKernel = self.blurringLayer.kernel[0, 0].to(device)
        paddingStyle = self.blurringLayer.padding
        ndims = len(shape)
        neighbours = [-1, +1]
        nNeighbours = len(neighbours)
        sz = [1 + ndims * nNeighbours, *shape]
        correlationMatrix = torch.zeros([1] + sz).to(device)
        if paddingStyle == 0:
            shape = torch.tensor(shape) + torch.tensor(smoothingKernel.shape) - 1
        onesTensor = torch.ones([1, 1, *shape]).to(device)
        # Get diagonal elements. Note that these are not 1 because of the weighing, therefore correlation matrix is not the correct word.
        # However, rescaling each of the elements to make the diagonal elements 1 would lead to the real correlation matrix. Diagonal
        # elements are stored as the first channel, which is optional for backwards compatibilty reasons
        squaredSmoothingKernel = (
            (smoothingKernel * smoothingKernel).unsqueeze(0).unsqueeze(0)
        )
        correlationMatrix[:, 0:1] = F.conv3d(
            onesTensor, squaredSmoothingKernel, padding=paddingStyle
        )

        # Get the neighbouring elements. For each dimension, we need the direct neighbours for the product with the adjacency matrix, others
        # are not necessary.
        for dim in range(ndims):
            for j, neighbour in enumerate(neighbours):
                index = dim * nNeighbours + j + 1
                squaredSmoothingKernel = self.getSquaredSmoothingKernel(
                    dim, neighbour, onesTensor.device
                )
                correlationMatrix[:, index : index + 1] = F.conv3d(
                    onesTensor, squaredSmoothingKernel, padding=paddingStyle
                )

        if self.rescaleCorrelationMatrix:
            correlationMatrix = correlationMatrix / correlationMatrix[:, 0:1]

        return correlationMatrix

    def logSigmaLoss(self, z, ndims):
        logSigma = z[:, ndims:]
        return torch.mean(logSigma)

    def sigmaTraceLoss(self, z, ndims):
        logSigma = z[:, ndims:]
        if self.D is None:
            self.D = self.getDegreeMatrix(z.shape[2:]).to(z.device)

        sigmaTraceTerm = self.D * torch.exp(logSigma)
        if self.sampleGaussianBlur:
            if self.K is None:
                self.K = self.getCorrelationMatrix(z.shape[2:], z.device)

            # TODO: here we need to account for the diagonal elements of the correlation matrix not being 1.
            # This can be alleviated with rescaling the correlation matrix. This does affect the loss function
            # because it changes the weight of this term compared to the other terms. This can then be
            # accounted for by changing lambda (prefactor of this term), but this would also change the
            # precision loss term. This means that it should not affect the case where the mean field is given,
            # but for the traditional voxelmorph this would change the results...

            sigmaTraceTerm = sigmaTraceTerm * self.K[:, 0:1]

            sig = torch.sqrt(torch.exp(logSigma))
            sigmaTraceTerm[:, :, 1:] -= (
                sig[:, :, :-1] * sig[:, :, 1:] * self.K[:, 1:2, 1:]
            )
            sigmaTraceTerm[:, :, :-1] -= (
                sig[:, :, :-1] * sig[:, :, 1:] * self.K[:, 2:3, :-1]
            )
            if ndims > 1:
                sigmaTraceTerm[:, :, :, 1:] -= (
                    sig[:, :, :, :-1] * sig[:, :, :, 1:] * self.K[:, 3:4, :, 1:]
                )
                sigmaTraceTerm[:, :, :, :-1] -= (
                    sig[:, :, :, :-1] * sig[:, :, :, 1:] * self.K[:, 4:5, :, :-1]
                )
            if ndims > 2:
                sigmaTraceTerm[:, :, :, :, 1:] -= (
                    sig[:, :, :, :, :-1]
                    * sig[:, :, :, :, 1:]
                    * self.K[:, 5:6, :, :, 1:]
                )
                sigmaTraceTerm[:, :, :, :, :-1] -= (
                    sig[:, :, :, :, :-1]
                    * sig[:, :, :, :, 1:]
                    * self.K[:, 6:7, :, :, :-1]
                )
        squaredVoxelSpacing = self.getSquaredVoxelSpacing(sigmaTraceTerm.device, ndims)
        return torch.mean(sigmaTraceTerm * self.lambdaPrior * squaredVoxelSpacing)

    def getSquaredVoxelSpacing(self, device, ndims):
        squaredVoxelSpacing = torch.square(
            torch.Tensor(self.fixedVoxelSpacing).to(device)
        )
        squaredVoxelSpacing = squaredVoxelSpacing.unsqueeze(0)
        for i in range(ndims):
            squaredVoxelSpacing = squaredVoxelSpacing.unsqueeze(-1)
        return squaredVoxelSpacing


class KLDivergenceProbabilisticBlockDiagonal(KLDivergenceProbabilisticDiagonal):
    def __init__(self, cfg):
        super().__init__(cfg)
        if self.fixedVoxelSpacing != (1.0, 1.0, 1.0):
            raise NotImplementedError

    def logSigmaLoss(self, z, ndims):
        choleskySquare = self.getSquaredCholeskyElements(z, ndims)
        diagonalIndices = [int((x + 1) * (x + 2) / 2) - 1 for x in range(ndims)]
        # for numerical stability add EPS
        return torch.log(choleskySquare[:, diagonalIndices] + EPS)

    def sigmaTraceLoss(self, z, ndims):
        choleskySquare = self.getSquaredCholeskyElements(z, ndims)
        covarianceElements = []
        for i in range(ndims):
            n = i + 1
            startIndex = int((n - 1) * (n) / 2)
            endIndex = startIndex + n
            covarianceElements.append(
                torch.sum(choleskySquare[:, startIndex:endIndex], dim=1, keepdim=True)
            )
        covarianceElements = torch.cat(covarianceElements, dim=1)
        return torch.mean(self.lambdaPrior * self.D * covarianceElements)

    def getSquaredCholeskyElements(self, z, ndims):
        choleskyLowerTriangular = z[:, ndims:]
        return torch.square(choleskyLowerTriangular)


