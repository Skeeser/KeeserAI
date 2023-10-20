import torch
from torch import nn
from d2l import torch as d2l
import Utils.LoadFashionData as ld

batch_size = 128
train_iter, test_iter = ld.load_data_fashion_mnist(batch_size=batch_size, resize=224)


class LeNet:
    def __init__(self):
        conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
        # 弄了个小的网络
        ratio = 4
        small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
        # self.net = self.vgg(conv_arch)
        self.net = self.vgg(small_conv_arch)

    def vgg_block(self, num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels,
                                    kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def vgg(self, conv_arch):
        conv_blks = []
        in_channels = 1
        # 卷积层部分
        for (num_convs, out_channels) in conv_arch:
            conv_blks.append(self.vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels

        return nn.Sequential(
            *conv_blks, nn.Flatten(),
            # 全连接层部分
            nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 10))


    def check(self):
        # （批量大小、通道、高度、宽度）
        X = torch.rand(size=(1, 1, 244, 244))
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape: \t', X.shape)

    def run(self):
        # self.check()
        lr, num_epochs, batch_size = 0.05, 10
        d2l.train_ch6(self.net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


if __name__ == "__main__":
    ln = LeNet()
    ln.run()