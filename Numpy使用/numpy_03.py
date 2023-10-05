import numpy as np

# 建立矩阵
# 建立一个大小为3 × 3的全 0 矩阵 c
c = np.zeros([3, 3], dtype=int)
print(c)

# 建立一个大小为4 × 5的全 1 矩阵 d
d = np.ones([4, 5], dtype=int)
print(d)

# 建立一个大小为4 × 4的单位矩阵 e
e = np.identity(4)
print(e)
