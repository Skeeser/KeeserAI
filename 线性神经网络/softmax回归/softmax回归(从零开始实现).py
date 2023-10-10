import torch
from IPython import display
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

# batch_size 256比较合理
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
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


train_iter, test_iter = load_data_fashion_mnist(batch_size)


class SoftMaxRegression:
    def __init__(self):
        # 初始化模型参数
        self.num_inputs = 784
        self.num_outputs = 10

        self.W = torch.normal(0, 0.01, size=(self.num_inputs, self.num_outputs), requires_grad=True)
        self.b = torch.zeros(self.num_outputs, requires_grad=True)

    def soft_max(self):
        pass


