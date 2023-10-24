import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from torch.utils import data
from torchvision import transforms
from Utils.Animator import Animator

batch_size = 256


def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4


def load_data_fashion_mnist(batch_size=256, resize=None):
    """下载MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.MNIST(
        root="../../resource", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.MNIST(
        root="../../resource", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


train_iter, test_iter = load_data_fashion_mnist(batch_size)


class MLP:
    def __init__(self):
        self.net = nn.Sequential(nn.Flatten(),
                            nn.Linear(784, 256),
                            nn.ReLU(),
                            nn.Linear(256, 10))

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, std=0.01)
        self.net.apply(init_weights)

    def train(self, net, train_iter, test_iter, loss, num_epochs, updater):
        animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                            legend=['train loss', 'train acc', 'test acc'])
        for epoch in range(num_epochs):
            train_metrics = d2l.train_epoch_ch3(net, train_iter, loss, updater)
            test_acc = d2l.evaluate_accuracy(net, test_iter)
            animator.add(epoch + 1, train_metrics + (test_acc,))
        train_loss, train_acc = train_metrics
        print(f'train_loss is: {train_loss}')
        print(f'train_acc is: {train_acc}')

    def run(self):
        batch_size, lr, num_epochs = 256, 0.01, 10
        loss = nn.CrossEntropyLoss(reduction='none')
        trainer = torch.optim.SGD(self.net.parameters(), lr=lr)

        self.train(self.net, train_iter, test_iter, loss, num_epochs, trainer)
        d2l.plt.show()


if __name__ == "__main__":
    mlp = MLP()
    mlp.run()