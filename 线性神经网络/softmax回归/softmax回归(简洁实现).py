import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from torch.utils import data
from torchvision import transforms
import threading


batch_size = 256


def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4


def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../../resource", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../../resource", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


train_iter, test_iter = load_data_fashion_mnist(batch_size)


class SoftMaxRegression:
    def __init__(self):
        # 初始化模型参数
        # PyTorch不会隐式地调整输入的形状。因此，
        # 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, std=0.01)
        self.net.apply(init_weights);

        self.loss = nn.CrossEntropyLoss(reduction='none')
        # 优化算法
        self.trainer = torch.optim.SGD(self.net.parameters(), lr=0.1)

    def run(self):
        num_epochs = 10
        d2l.train_ch3(self.net, train_iter, test_iter, self.loss, num_epochs, self.trainer)
        d2l.plt.show()  # 显示绘图


if __name__ == "__main__":
    SMR = SoftMaxRegression()
    SMR.run()
