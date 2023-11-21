import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt


"""
自编码器
现在自动编码器主要应用在两个方面：第一是数据去噪，第二是进行可视化降维。
自动编码器还有一个功能，即生成数据。
CNN
"""

batch_size = 128


# 加载数据集
def get_data():
    # 将像素点转换到[-1, 1]之间，使得输入变成一个比较对称的分布，训练容易收敛
    data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_dataset = datasets.MNIST(root="../resource", train=True, transform=data_tf, download=True)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    return train_loader


def to_img(x):
    x = (x + 1.) * 0.5
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


train_iter = get_data()


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # 初始化encoder和decoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # (b, 16, 10, 10)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # (b, 16, 5, 5)
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # (b, 8, 3, 3)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # (b, 8, 2, 2)
        )

        # 注意此处decoder用到了反卷积
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # (b, 16, 5, 5)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # (b, 8, 15, 15)
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # (b, 1, 28, 28)
            nn.Tanh()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

    def train_sl(self,train_iter, num_epochs, lr, weight_decay):
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
        epoches = 100
        self.train_sl(train_iter, epoches, lr, weight_decay)


if __name__ == "__main__":
    model = AutoEncoder()
    model.run()

    torch.save(model, '../model/autoencoder.pth')
    code = Variable(torch.FloatTensor([[1.19, -3.36, 2.06]]).cuda())
    decode = model.decoder(code)
    decode_img = to_img(decode).squeeze()
    decode_img = decode_img.data.cpu().data * 255
    plt.imshow(decode_img.astype('uint8'), cmap='gray')
    save_image(decode_img, '../resource/simple_autoencoder/image_code.png')
    plt.show()
