import os
import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.utils import save_image
from d2l import torch as d2l


# 加载数据集
def get_data():
    data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_data = datasets.MNIST(root="../resource", train=True, transform=data_tf, download=True)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    return train_loader


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # self.fc1 = nn.Linear(784, 400)
        self.fc1 = nn.Sequential(nn.Linear(784, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 512),
                                     nn.Tanh())


        self.fc21 = nn.Linear(512, 20)    # 均值
        self.fc22 = nn.Linear(512, 20)    # 方差
        self.fc3 = nn.Sequential(nn.Linear(20, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 784),
                                     nn.Tanh())


    def encoder(self, x):
        h1 = self.fc1(x)
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar

    def decoder(self, z):
        x = self.fc3(z)
        return x

    # 重新参数化
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()   # 计算标准差
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()    # 从标准的正态分布中随机采样一个eps
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        return self.decoder(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    MSE = reconstruction_function(recon_x, x)
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # print(f"MSE: {MSE}, KLD: {KLD}")
    # KL divergence
    return MSE + KLD, MSE, KLD


def to_img(x):
    x = x[0:show_size]
    x = (x + 1.) * 0.5
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


if __name__ == '__main__':
    # 超参数设置
    batch_size = 128
    show_size = 10
    lr = 1e-3
    epoches = 100

    model = VAE()
    if torch.cuda.is_available():
        model.cuda()

    train_data = get_data()

    reconstruction_function = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    animator1 = d2l.Animator(xlabel='epoch', xlim=[1, epoches],
                            legend=['MSE'])
    animator2 = d2l.Animator(xlabel='epoch', xlim=[1, epoches],
                            legend=['KLD'])
    num_batches = len(train_data)
    for epoch in range(epoches):
        # 在训练过程中逐渐降低学习率，以便在接近训练结束时更细致地调整模型参数，使得模型更容易收敛到最优解。
        if epoch in [epoches * 0.25, epoches * 0.5]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        for i, (img, _) in enumerate(train_data):
            img = img.view(img.size(0), -1)
            img = Variable(img)
            if torch.cuda.is_available():
                img = img.cuda()
            # forward
            output, mu, logvar = model(img)
            loss_l = loss_function(output, img, mu, logvar)
            loss = loss_l[0]/img.size(0)
            mse = loss_l[1]/img.size(0)
            kld = loss_l[2]/img.size(0)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator1.add(epoch + (i + 1) / num_batches, (mse.cpu().data, None))
                animator2.add(epoch + (i + 1) / num_batches, (kld.cpu().data, None))
        print("epoch=", epoch, loss.data.float())
        if (epoch+1) % 10 == 0:
            print("epoch = {}, loss is {}".format(epoch+1, loss.data))
            pic = to_img(output.cpu().data)
            if not os.path.exists('../resource/vae_img1'):
                os.mkdir('../resource/vae_img1')
            save_image(pic, '../resource/vae_img1/image_{}.png'.format(epoch + 1))
    # torch.save(model, '../model/vae.pth')
    d2l.plt.show()