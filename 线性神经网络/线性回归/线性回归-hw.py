import numpy as np
import matplotlib.pyplot as plt
import torch
from Utils.DataIter import DataIter

# 生成训练数据，数据为带有服从-0.5 到 0.5 的均匀分布噪声的正弦函数
num_observations = 100
np.random.seed(0)
noise = np.random.uniform(-0.5, 0.5, num_observations)

x = np.linspace(-3, 3, num_observations)
y = np.sin(x) + noise

features = torch.from_numpy(x.reshape((-1, 1))).float()
labels = torch.from_numpy(y.reshape(-1, 1)).float()

# 画出这 100 个样本的散点图。（提交散点图）
# plt.scatter(x, y)
# plt.show()

# 读取数据集
batch_size = 100


class LinearRegression:
    def __init__(self):
        # 初始化参数, 并启用梯度跟踪
        # 梯度跟踪能够自动微分
        self.w = torch.zeros(1, requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    # 定义模型
    def line_reg(self, X, w, b):
        """线性回归模型"""
        return torch.matmul(X, w) + b

    # 定义损失函数
    def squared_loss(self, y_hat, y):
        """均方损失"""
        return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

    # 定义优化算法, 参数 模型参数集合、学习速率(步长)和批量大小
    def sgd(self, params, lr, batch_size):
        """小批量随机梯度下降"""
        # 在一个上下文中禁用梯度计算
        # 可以避免不必要的梯度计算，从而提高代码的效率并减少内存消耗。
        with torch.no_grad():
            for param in params:
                param -= lr * param.grad / batch_size
                # 清空参数的梯度
                param.grad.zero_()

    # 训练！！！
    def train(self, lr=0.03, num_epochs=10):
        #  这里的迭代周期个数num_epochs和学习率lr都是超参数，分别设为3和0.03
        # lr = 0.03
        # num_epochs = 3
        net = self.line_reg
        loss = self.squared_loss

        for epoch in range(num_epochs):
            for X, y in DataIter.data_iter(batch_size, features, labels):
                l = loss(net(X, self.w, self.b), y)  # X和y的小批量损失
                # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
                # 并以此计算关于[w,b]的梯度
                # 调用 .backward() 方法，这个方法用于自动计算这个标量损失对模型参数的梯度。
                # 它会通过自动微分（autograd）机制追踪计算图中的操作，并计算损失相对于每个参数的梯度。
                l.sum().backward()
                self.sgd([self.w, self.b], lr, batch_size)  # 使用参数的梯度更新参数
            with torch.no_grad():
                train_l = loss(net(features, self.w, self.b), labels)
                print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

        print(f'w的训练值: {self.w}')
        print(f'b的训练值: {self.b}')

    def draw(self):
        plt.scatter(x, y)
        plt.plot(features.reshape(x.shape).detach().numpy(), (labels.reshape(y.shape) + self.line_reg(features, self.w, self.b).reshape(y.shape)).detach().numpy(), color='red', linewidth=2)
        plt.show()


if __name__ == "__main__":
    lg = LinearRegression()
    lg.train(num_epochs=50)
    lg.draw()

