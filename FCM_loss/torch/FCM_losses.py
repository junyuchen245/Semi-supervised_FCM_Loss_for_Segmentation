from torch import nn
import torch
import math
from torch.nn import functional as F

class FCM_label(nn.Module):
    def __init__(self, fuzzy_factor=2, one_hot=True):
        '''
        Supervised Fuzzy C-mean loss function for ConvNet based image segmentation
        Junyu Chen, et al. Learning Fuzzy Clustering for SPECT/CT Segmentation
        via Convolutional Neural Networks. Medical physics, 2021 (In press).
        :param fuzzy_factor: exponent for controlling fuzzy overlap, default value = 2
        :param one_hot: setting to True will convert truth into one hot encoding
        :param y_pred: prediction from ConvNet, assuming that SoftMax has been applied.
        :param y_true: ground truth label
        '''
        super().__init__()
        self.fuzzy_factor = fuzzy_factor
        self.one_hot = one_hot

    def forward(self, y_pred, y_true):
        B, C, W, H, L = y_pred.shape
        dim = len(list(y_pred.shape)[2:])
        assert dim == 3 or dim == 2, 'Supports only 3D or 2D!'
        num_clus = C
        if self.one_hot:
            y_true = F.one_hot(y_true.long(), num_classes=num_clus)
            y_true = torch.squeeze(y_true, 1)
            if dim == 3:
                y_true = y_true.permute(0, 4, 1, 2, 3)
            else:
                y_true = y_true.permute(0, 3, 1, 2)
        pred = torch.reshape(y_pred, (B, num_clus, math.prod(list(y_true.shape)[2:])))
        truth = torch.reshape(y_true, (B, num_clus, math.prod(list(y_true.shape)[2:])))
        J_1 = 0
        for i in range(num_clus):
            label = truth[:, i, :]
            mem = torch.pow(pred[:, i, :], self.fuzzy_factor)
            J_1 += mem*torch.pow(label - 1, 2)
        return torch.mean(J_1/num_clus)

class RFCM_loss(nn.Module):
    def __init__(self, fuzzy_factor=2, regularizer_wt=0.0008):
        '''
        Unsupervised Robust Fuzzy C-mean loss function for ConvNet based image segmentation
        Junyu Chen, et al. Learning Fuzzy Clustering for SPECT/CT Segmentation
        via Convolutional Neural Networks. Medical physics, 2021 (In press).
        :param fuzzy_factor: exponent for controlling fuzzy overlap, default value = 2
        :param regularizer_wt: weighting parameter for regularization, default value = 0
        Note that ground truth segmentation is NOT needed in this loss fuction, instead, the input image is required.
        :param y_pred: prediction from ConvNet, assuming that SoftMax has been applied.
        :param image: input image to the ConvNet.
        '''
        super().__init__()
        self.fuzzy_factor = fuzzy_factor
        self.wt = regularizer_wt


    def forward(self, y_pred, image):
        dim = len(list(y_pred.shape)[2:])
        assert dim == 3 or dim == 2, 'Supports only 3D or 2D!'
        num_clus = y_pred.shape[1]
        pred = torch.reshape(y_pred, (y_pred.shape[0], num_clus, math.prod(list(y_pred.shape)[2:]))) #(bs, C, V)
        img = torch.reshape(image, (y_pred.shape[0], math.prod(list(image.shape)[2:]))) #(bs, V)
        if dim == 3:
            kernel = torch.ones((1, 1, 3, 3, 3)).float().cuda()
            kernel[:, :, 1, 1, 1] = 0
        else:
            kernel = torch.ones((1, 1, 3, 3)).float().cuda()
            kernel[:, :, 1, 1] = 0

        J_1 = 0
        J_2 = 0
        for i in range(num_clus):
            mem = torch.pow(pred[:, i, ...], self.fuzzy_factor) #extracting membership function (bs, V)
            v_k = torch.sum(img * mem, dim=1, keepdim=True)/torch.sum(mem, dim=1, keepdim=True) #scalar
            J_1 += mem*torch.square(img - v_k) #(bs, V)
            J_in = 0
            for j in range(num_clus):
                if i==j:
                    continue
                mem_j = torch.pow(pred[:, j, ...], self.fuzzy_factor)
                mem_j = torch.reshape(mem_j, image.shape)
                if dim == 3:
                    res = F.conv3d(mem_j, kernel, padding=int(3 / 2))
                else:
                    res = F.conv2d(mem_j, kernel, padding=int(3 / 2))
                res = torch.reshape(res, (-1, math.prod(list(image.shape)[2:])))
                J_in += res #(bs, V)
            J_2 += mem * J_in #(bs, V)
        return torch.mean(J_1)+self.wt*torch.mean(J_2)
