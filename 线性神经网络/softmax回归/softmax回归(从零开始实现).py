import torch

import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from Utils.Accumulator import Accumulator
from Utils.Animator import Animator

# batch_size 256比较合理
batch_size = 256
"""!!!!!有bug"""

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
        self.num_inputs = 784
        self.num_outputs = 10

        self.W = torch.normal(0, 0.01, size=(self.num_inputs, self.num_outputs), requires_grad=True)
        self.b = torch.zeros(self.num_outputs, requires_grad=True)

    def soft_max(self, X):
        X_exp = torch.exp(X)
        partition = X_exp.sum(1, keepdim=True)
        return X_exp / partition

    def net(self, X):
        return self.soft_max(torch.matmul(X.reshape((-1, self.W.shape[0])), self.W) + self.b)

    # 输入预测概率分布和对应标签
    def cross_entropy(self, y_hat, y):
        return - torch.log(y_hat[range(len(y_hat)), y])

    def accuracy(self, y_hat, y):
        """计算预测正确的数量"""
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(axis=1)
        cmp = y_hat.type(y.dtype) == y
        return float(cmp.type(y.dtype).sum())

    def evaluate_accuracy(self, net, data_iter):
        """计算在指定数据集上模型的精度"""
        if isinstance(net, torch.nn.Module):
            net.eval()  # 将模型设置为评估模式
        metric = Accumulator(2)  # 正确预测数、预测总数
        with torch.no_grad():
            for X, y in data_iter:
                metric.add(self.accuracy(net(X), y), y.numel())
        return metric[0] / metric[1]

    def train_epoch(self, net, train_iter, loss, updater):
        """训练模型的一个迭代周期"""
        """updater--优化函数，常见的比如梯度下降算法"""
        if isinstance(net, torch.nn.Module):
            net.train()
        # 累加器, 分别为训练损失总和、训练准确度总和、样本数
        metric = Accumulator(3)
        for X, y in train_iter:
            # 计算梯度并更新参数
            y_hat = net(X)
            l = loss(y_hat, y)
            print(self.W)
            if isinstance(updater, torch.optim.Optimizer):
                # 使用pytorch内置的优化器和损失函数
                updater.zero_grad()
                l.mean().backward()
                # 更新权重
                updater.step()
            else:
                # 使用定制的优化器和损失函数
                l.sum().backward()
                updater(X.shape[0])
            metric.add(float(l.sum()), self.accuracy(y_hat, y), y.numel())
        # 返回训练损失和训练精度
        return metric[0] / metric[2], metric[1] / metric[2]

    def train(self, net, train_iter, test_iter, loss, num_epochs, updater):
        """训练模型"""
        # 定义绘图
        animator = Animator(xlabel='epoch', legend=['train loss', 'train acc', 'test acc'], xlim=[1, num_epochs], ylim=[0.3, 0.9])
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(net, train_iter, loss, updater)
            test_acc = self.evaluate_accuracy(net, test_iter)
            animator.add(epoch + 1, train_metrics + (test_acc,))
            train_loss, train_acc = train_metrics
            assert train_loss < 0.5, train_loss
            assert train_acc <= 1 and train_acc > 0.7, train_acc
            assert test_acc <= 1 and test_acc > 0.7, test_acc

    def updater(self, lr=0.1, batch_size=256):
        return d2l.sgd([self.W, self.b], lr, batch_size)

    def predict(self, net, test_iter, n=6):
        """预测标签"""
        for X, y in test_iter:
            break
        trues = d2l.get_fashion_mnist_labels(y)
        preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
        titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
        d2l.show_images(
            X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

    def run(self):
        num_epochs = 10
        self.train(self.net, train_iter, test_iter, self.cross_entropy, num_epochs, self.updater)
        self.predict(self.net, test_iter)


if __name__ == "__main__":
    SMR = SoftMaxRegression()
    SMR.run()

