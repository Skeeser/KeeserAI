import torch
from torch import nn
from d2l import torch as d2l

# 实现二维卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def corr2d(self, X, K):
        """计算二维互相关计算"""
        h, w = K.shape
        Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
        return Y

    def forward(self, x):
        return self.corr2d(x, self.weight) + self.bias


class CNN:
    def __init__(self):
        pass



