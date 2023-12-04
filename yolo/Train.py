import numpy as np
from dataset import yoloDataset
from yoloLoss import yoloLoss
from yolo_v1_net import resnet50
from torch.autograd import Variable
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import os
import argparse


device = 'cuda' if torch.cuda.is_available() else 'cpu'
file_root = 'F:/face_datas/VOC/VOCdevkit/VOC2012/JPEGImages/'
batch_size = 2
learning_rate = 0.001
num_epochs = 30

train_dataset = yoloDataset(
    root=file_root, list_file=
        'voc2012.txt', train=True, transform=[
            transforms.ToTensor()])
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0)
test_dataset = yoloDataset(
    root=file_root,
    list_file='voc2007test.txt',
    train=False,
    transform=[
        transforms.ToTensor()])
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0)
print('the dataset has %d images' % (len(train_dataset)))
print('the batch_size is %d' % (batch_size))
print('loading network structure...')

net = resnet50()
net = net.to(device)
#print(net)

print('load pre_trained model...')
resnet = models.resnet50(pretrained=True)
new_state_dict = resnet.state_dict()

op = net.state_dict()
for k in new_state_dict.keys():
    print(k)
    if k in op.keys() and not k.startswith('fc'):  # startswith() 方法用于检查字符串是否是以指定子字符串开头，如果是则返回 True，否则返回 False
        print('yes')
        op[k] = new_state_dict[k]
net.load_state_dict(op)

if False:   # 这个地方不是特别懂，下面再看看其他的理解了再修改
    net.load_state_dict(torch.load('best.pth'))
print('testing the cuda device here')
print('cuda', torch.cuda.current_device(), torch.cuda.device_count())

criterion = yoloLoss(7, 2, 5, 0.5)

net.train()
# different learning rate
params = []
params_dict = dict(net.named_parameters())

for key, value in params_dict.items():
    if key.startswith('features'):
        params += [{'params': [value], 'lr':learning_rate * 1}]  # 这儿的学习率好像并没有变化？？？
    else:
        params += [{'params': [value], 'lr':learning_rate}]
optimizer = torch.optim.SGD(
    params,
    lr=learning_rate,
    momentum=0.9,
    weight_decay=5e-4)

torch.multiprocessing.freeze_support()

for epoch in range(num_epochs):
    net.train()
    if epoch == 30:
        learning_rate = 0.0001
    if epoch == 40:
        learning_rate = 0.00001
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
    print('Learning Rate for this epoch: {}'.format(learning_rate))

    total_loss = 0.
    for i, (images, target) in enumerate(train_loader):
        images, target = images.cuda(), target.cuda()
        pred = net(images)
        loss = criterion(pred, target)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 5 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' % (epoch +1, num_epochs,
                                                                                 i + 1, len(train_loader), loss.item(), total_loss / (i + 1)))
    validation_loss = 0.0
    net.eval()
    for i, (images, target) in enumerate(test_loader):
        images, target = images.cuda(), target.cuda()
        pred = net(images)
        loss = criterion(pred, target)
        validation_loss += loss.item()
    validation_loss /= len(test_loader)

    if best_test_loss > validation_loss:
        best_test_loss = validation_loss
        print('get best test loss %.5f' % best_test_loss)
        torch.save(net.state_dict(), 'best.pth')
    torch.save(net.state_dict(), 'yolo.pth')