import torch
from torch import nn

# 生成标准正态分布随机数
# 参数是 a sequence of integers defining the shape of the output tensor.
parameter = nn.Parameter(torch.randn())

# 保存参数
# 其中net为自定义的模型名称，其子参数state_dict()为模型的参数，path为保存的路径加名称，其后缀为 pt 或 pth ，
# 如： ‘pth/net_parameters.pth’。
torch.save(net.state_dict(), path)

# 加载参数
net.load_state_dict(torch.load(path))

# 保存和读取模型
torch.save(net, path) # 保存模型
net_ = torch.load(pth) # 读取模型