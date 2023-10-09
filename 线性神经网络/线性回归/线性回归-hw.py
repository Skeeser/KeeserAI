import numpy as np
import matplotlib.pyplot as plt

# 生成训练数据，数据为带有服从-0.5 到 0.5 的均匀分布噪声的正弦函数
np.random.seed(0)
noise = np.random.uniform(-0.5, 0.5, 100)

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x) + noise

# 画出这 100 个样本的散点图。（提交散点图）
# plt.scatter(x, y)
# plt.show()

# 使用 Pytorch/Tensorflow 实现线性回归模型，训练参数w和b。 𝑦 = 𝑤 ∗ 𝑥 #
# 输出参数 w、b 和 损失。
# 画出预测回归曲线以及训练数据散点图，对比回归曲线和散点图并分析原因。
