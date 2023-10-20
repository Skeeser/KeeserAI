import torch
from torch import nn
from d2l import torch as d2l
import Utils.LoadFashionData as ld

batch_size = 256
train_iter, test_iter = ld.load_data_fashion_mnist(batch_size=batch_size)


class LeNet:
    def __init__(self):
        # padding = 2表示填充两个像素 stride=2表示步幅为两个像素
        # 增加通道的数量对应多个卷积核，多个特征
        # 假设输入28*28
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),  # 6@28*28
            nn.AvgPool2d(kernel_size=2, stride=2),  # 6@14*14
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),  # 16@10*10
            nn.AvgPool2d(kernel_size=2, stride=2),  # 16@5*5
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def check(self):
        # （批量大小、通道、高度、宽度）
        X = torch.rand(size=(1, 1, 28, 18), dtype=torch.float32)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape: \t', X.shape)

    def run(self):
        self.check()


if __name__ == "__main__":
    ln = LeNet()
    ln.run()