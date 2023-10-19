import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from torch.utils import data
from torchvision import transforms

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


class MLP:
    def __init__(self):
        # 初始化模型参数
        self.num_inputs, self.num_outputs, self.num_hiddens = 784, 10, 256

        self.W1 = nn.Parameter(torch.randn(self.num_inputs, self.num_hiddens, requires_grad=True) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(self.num_hiddens, requires_grad=True))
        self.W2 = nn.Parameter(torch.randn(self.num_hiddens, self.num_outputs, requires_grad=True) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(self.num_outputs, requires_grad=True))

        self.params = [self.W1, self.b1, self.W2, self.b2]

        self.loss = nn.CrossEntropyLoss(reduction='none')

    # 激活函数
    def relu(self, X):
        # 弄成0
        a = torch.zeros_like(X)
        return torch.max(X, a)

    def net(self, X):
        X = X.reshape((-1, self.num_inputs))
        H = self.relu(X@self.W1 + self.b1)  # @是矩阵乘法
        return (H@self.W2 + self.b2)

    def train(self):
        num_epochs, lr = 10, 0.1
        updater = torch.optim.SGD(self.params, lr=lr)
        d2l.train_ch3(self.net, train_iter, test_iter, self.loss, num_epochs, updater)

    def run(self):
        self.train()
        d2l.predict_ch3(self.net, test_iter)
        d2l.plt.show()


if __name__ == "__main__":
    mlp = MLP()
    mlp.run()