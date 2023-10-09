import random
import torch
from d2l import torch as d2l
from Utils.DataIter import DataIter


# 生成数据集
def synthetic_data(w, b, num_example):
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_example, len(w)))
    # 矩阵乘法
    y = torch.matmul(X, w) + b
    # 加上噪声
    y += torch.normal(0, 0.01, y.shape)
    # y.reshape((-1, 1))重新塑造成二维张量，只有一列
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# features中的每一行都包含一个二维数据样本，
# labels中的每一行都包含一维标签值（一个标量）。

d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
# d2l.plt.show()

# 读取数据集
batch_size = 10

# for X, y in DataIter.data_iter(batch_size, features, labels):
#     print(X, '\n', y)
#     break


class LinearRegression:
    def __init__(self):
        # 初始化参数, 并启用梯度跟踪
        # 梯度跟踪能够自动微分
        self.w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
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
    def train(self, lr=0.03, num_epochs=3):
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

        print(f'w的估计误差: {true_w - self.w.reshape(true_w.shape)}')
        print(f'b的估计误差: {true_b - self.b}')


if __name__ == "__main__":
    lg = LinearRegression()
    lg.train()

