import numpy as np

# 建立一个二维数组 b，初始化为[[4,5,6], [1,2,3]]
b = np.array([[4, 5, 6], [1, 2, 3]])
print(b)

# 输出 b 的各维度大小（shape）
print(b.shape)

# 输出b[0,0], b[0,1], b[1,1]这三个元素
print(b[0, 0], b[0, 1], b[1, 1])
