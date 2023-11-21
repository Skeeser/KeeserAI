import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
import Utils.LoadFashionData as load_data
from d2l import torch as d2l

"""
自编码器
现在自动编码器主要应用在两个方面：第一是数据去噪，第二是进行可视化降维。
自动编码器还有一个功能，即生成数据。
全连接网络
"""

batch_size = 128

train_iter, test_iter = load_data.load_data_mnist(batch_size)

def to_img(x):
    x = (x + 1.) * 0.5
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # 初始化encoder和decoder
        self.encoder = nn.Sequential(nn.Linear(28*28, 128),
                                     nn.ReLU(True),
                                     nn.Linear(128, 64),
                                     nn.ReLU(True),
                                     nn.Linear(64, 12),
                                     nn.ReLU(True),
                                     nn.Linear(12, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 12),
                                     nn.ReLU(True),
                                     nn.Linear(12, 64),
                                     nn.ReLU(True),
                                     nn.Linear(64, 128),
                                     nn.ReLU(True),
                                     nn.Linear(128, 28*28),
                                     nn.Tanh())

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

    def train_sl(self,train_iter, test_iter, num_epochs, lr, weight_decay):
        # 均方损失
        loss = nn.MSELoss()
        # 优化器adam  参数学习率和权重衰减
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        if torch.cuda.is_available():
            self.cuda()
        for epoch in range(num_epochs):
            # 在训练过程中逐渐降低学习率，以便在接近训练结束时更细致地调整模型参数，使得模型更容易收敛到最优解。
            if epoch in [num_epochs * 0.25, num_epochs * 0.5]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
            for i, (img, _) in enumerate(train_iter):
                img = img.view(img.size(0), -1)
                img = Variable(img.cuda())
                # forward
                _, output = self.forward(img)
                l = loss(output, img)
                # backward
                optimizer.zero_grad()
                l.backward()
                optimizer.step()

            print("epoch=", epoch, l.data.float())
            for param_group in optimizer.param_groups:
                print(param_group['lr'])

            if (epoch + 1) % 5 == 0:
                print("epoch: {}, loss is {}".format((epoch + 1), l.data))
                pic = to_img(output.cpu().data)
                if not os.path.exists('../resource/simple_autoencoder'):
                    os.mkdir('../resource/simple_autoencoder')
                save_image(pic, '../resource/simple_autoencoder/image_{}.png'.format(epoch + 1))

    def run(self):
        # 超参数设置
        lr = 1e-2
        weight_decay = 1e-5
        epoches = 40
        self.train_sl(train_iter, test_iter, epoches, lr, weight_decay)

        code = Variable(torch.FloatTensor([[1.19, -3.36, 2.06]]).cuda())
        decode = self.decoder(code)
        decode_img = to_img(decode).squeeze()
        decode_img = decode_img.data.cpu().numpy() * 255
        plt.imshow(decode_img.astype('uint8'), cmap='gray')
        plt.show()


if __name__ == "__main__":
    model = AutoEncoder()
    model.run()
