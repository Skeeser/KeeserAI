import numpy as np
import torch

# 从NumPy数组到PyTorch张量的转换：
# 创建一个NumPy数组
numpy_array = np.array([1, 2, 3])

# 将NumPy数组转换为PyTorch张量
torch_tensor = torch.from_numpy(numpy_array)

# 可以选择将NumPy数组的数据类型转换为匹配的PyTorch数据类型
torch_tensor = torch.from_numpy(numpy_array).float()


# 从PyTorch张量到NumPy数组的转换：
# 创建一个PyTorch张量
torch_tensor = torch.tensor([1.0, 2.0, 3.0])

# 将PyTorch张量转换为NumPy数组
# 如果自己在转换数据时不需要保留梯度信息，可以在变量转换之前添加detach()调用。
numpy_array = torch_tensor.numpy()