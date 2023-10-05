import numpy as np

# 建立一个数组 f，初始化为[0,1,2,3,4,5,6,7,8,9,10,11]（arange）
f = np.arange(0, 12)

# 输出 f 以及 f 的各维度大小
print(f)
print(f.shape)

# 将 f 的 shape 改为3 × 4（reshape）
f = f.reshape(3, 4)

# 输出 f 以及 f 的各维度大小
print(f)
print(f.shape)

# 输出 f 第二行（f[1, ∶]）
print(f[1, :])

# 输出 f 最后两列（f[: ,2: ]）
print(f[:, 2:])

# 输出 f 第三行最后一个元素（使用-1 表示最后一个元素）
print(f[2, -1])
