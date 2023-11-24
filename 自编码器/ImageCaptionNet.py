import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import json
from sklearn.utils import shuffle
from PIL import Image, ImageOps
import torchtext
from sklearn.model_selection import train_test_split
from d2l import torch as d2l
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

if torch.cuda.is_available():
    print("cuda")
    device = torch.device("cuda:0")
else:
    print("cpu")
    device = torch.device("cpu")

# 定义批处理大小
batch = 8
# 定义最大词汇表容量
max_token = 5000
# 定义句子的最大长度
MAX_LEN = 30
# 定义自动填充内容
PAD = 0
# 定义图片特征向量
img_feature = 256


# 自定义一个类，用于从 LSTM 中提取最后一个时间步的输出
class LastTimeStepExtractor(nn.Module):
    def forward(self, x):
        a, (_, _) = x
        return a[0]


# 实现看图说话模型
class ImageCaption(nn.Module):
    def __init__(self):
        super(ImageCaption, self).__init__()
        resnet = models.resnet152()
        resnet.fc = nn.Linear(2048, img_feature)
        self.device = device
        # 编码器是CNN网络
        self.encoder = nn.Sequential(resnet,
                                     nn.BatchNorm1d(img_feature, momentum=0.01)).to(device)  # 归一正则化

        # 解码器是LSTM网络
        self.decoder = nn.LSTM(input_size=img_feature, hidden_size=512, num_layers=1, batch_first=True).to(device)
        self.linear = nn.Linear(512, max_token).to(device)
        self.dropout = nn.Dropout(p=0.1).to(device)  # 防止过拟合

        # 编码
        self.embedding = nn.Embedding(max_token, img_feature).to(device)

    def encoder_func(self, x):
        z = self.encoder(x)
        return z

    def decoder_func(self, features, y, lengths):
        # 对给定词序列进行编码
        y = self.embedding(y)
        use = []
        for i in range(len(y)):
            # 注意图片特征向量在前面,因为其要作为LSTM层的初始状态
            temp = torch.cat((features[i].unsqueeze(0), y[i]), dim=0)
            use.append(temp)
        # 结果为列表,利用stack函数将其转变为tensor,第一维为批处理大小
        use = torch.stack(use, dim=0)
        use = use.to(device)
        x = pack_padded_sequence(use, lengths, batch_first=True)
        hidden, _ = self.decoder(x)
        y_pred = self.linear(hidden[0])
        # y_pred = self.dropout(y_pred)
        return y_pred

    def infer_sl(self, features, states=None):
        res = []
        inputs = features.unsqueeze(1)
        for i in range(MAX_LEN):
            hidden, states = self.decoder(inputs, states)
            outputs = self.linear(hidden.squeeze(1))
            _, predicted = outputs.max(1)
            res.append(predicted)
            inputs = self.embedding(predicted)
            inputs = inputs.unsqueeze(1)

        # 列表转tensor
        res = torch.stack(res, 1)
        return res

    def forward(self, x, y, lengths):
        feature = self.encoder_func(x)
        y_pred = self.decoder_func(feature, y, lengths)
        return y_pred


