import torch
from torch import nn

# 生成标准正态分布随机数
# 参数是 a sequence of integers defining the shape of the output tensor.
parameter = nn.Parameter(torch.randn())