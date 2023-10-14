import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
# nn是神经网络的缩写
from torch import nn

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)

# print(next(iter(data_iter)))


class LinearRegression:
    def __init__(self):
        # 定义Sequential类对象 Sequential类将多个层串联在一起。
        # 线性回归只有一层
        self.net = nn.Sequential(nn.Linear(2, 1))
        # 初始化模型参数，即w和b
        self.net[0].weight.data.normal_(0, 0.01)
        self.net[0].bias.data.fill_(0)

        # 定义损失函数
        self.loss = nn.MSELoss()

        # 定义优化算法 小批量随机梯度下降算法
        self.trainer = torch.optim.SGD(self.net.parameters(), lr=0.3)

    def train(self, num_epochs = 3):
        for epoch in range(num_epochs):
            for X, y in data_iter:
                l = self.loss(self.net(X), y)
                # 清零梯度
                self.trainer.zero_grad()
                # 反向传播计算梯度
                l.backward()
                # 更新模型参数
                self.trainer.step()
            l = self.loss(self.net(features), labels)
            print(f'epoch {epoch + 1}, loss {l:f}')

            w = self.net[0].weight.data
            print('w的估计误差：', true_w - w.reshape(true_w.shape))
            b = self.net[0].bias.data
            print('b的估计误差：', true_b - b)


if __name__ == "__main__":
    lg = LinearRegression()
    lg.train()

