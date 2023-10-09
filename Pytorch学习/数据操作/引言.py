import torch

x = torch.arange(12)

print(x)
print(x.shape)
print(x.numel())
print(x.reshape(3, 4))
print(torch.zeros(2, 3, 4))
print(torch.ones(2, 3, 4))

# 创建（3,4）的张量，
# 并且每个元素满足均值0，标准差1的高斯分布（正态分布）随机采样
print(torch.randn(3, 4))

print(torch.Tensor([[1, 2], [3, 4]]))

