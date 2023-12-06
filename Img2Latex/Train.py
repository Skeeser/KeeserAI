from __future__ import division

import os
import random
import argparse
import time
import math
import numpy as np
import cv2
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from Data import VOCDetection
import Tools
from Model import myYOLO
from Utils import SSDAugmentation
# from utils.vocapi_evaluator import VOCAPIEvaluator


# 数据预处理函数
def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


train_cfg = {
    'lr_epoch': (20, 40, 80),
    'max_epoch': 80,
    'min_dim': [416, 416]
}

CLASSES = (  # always index 0
    'math')


def train():
    # 设置保存地址
    path_to_save = "./out"
    os.makedirs(path_to_save, exist_ok=True)

    # 不使用高分辨率的骨干网络
    hr = False

    # cuda
    if torch.cuda.is_available():
        print("cuda")
        device = torch.device("cuda:0")
    else:
        print("cpu")
        device = torch.device("cpu")

    # 多尺度处理
    multi_scale = False
    if multi_scale:
        print('use the multi-scale trick ...')
        train_size = [640, 640]
        val_size = [416, 416]
    else:
        train_size = [416, 416]
        val_size = [416, 416]


    # 加载数据集
    print("加载数据集......")
    # 设置数据集地址
    data_dir = "../resource/yolo"
    num_classes = 1
    dataset = VOCDetection(root=data_dir,
                           img_size=train_size[0],
                           transform=SSDAugmentation(train_size)
                           )

    # evaluator = VOCAPIEvaluator(data_root=data_dir,
    #                             img_size=val_size,
    #                             device=device,
    #                             transform=BaseTransform(val_size),
    #                             labelmap=CLASSES
    #                             )



    # dataloader
    batch_size = 16
    # 用于数据加载的子进程数量
    num_workers = 8
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=detection_collate,
        num_workers=num_workers,
        pin_memory=True
    )

    # 加载模型
    print("加载模型......")
    yolo_net = myYOLO(device, input_size=train_size, num_classes=num_classes, trainable=True)
    model = yolo_net
    model.to(device).train()

    # 是否使用tfboard记录和可视化训练过程中的数据
    tensorboard = True
    if tensorboard:
        print('使用tensorboard可视化......')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime(time.time()))
        log_path = os.path.join('./log/', c_time)
        os.makedirs(log_path, exist_ok=True)
        writer = SummaryWriter(log_path)

    # 是否要继续之前的模型, 要的话改为之前训练的模型地址
    resume = None
    if resume is not None:
        print('继续训练....模型地址为: %s' % (resume))
        model.load_state_dict(torch.load(resume, map_location=device))

    # 超参数设置
    # 学习率, 优化器
    base_lr = 1e-4
    tmp_lr = base_lr
    # momentum是动量（momentum）的概念，它在更新参数时考虑了之前更新的方向，
    # 有助于加速收敛并且有助于在局部最小值周围更快地搜索到全局最小值。
    momentum = 0.9
    # L2正则化项, 防止过拟合
    weight_decay = 5e-4
    optimizer = optim.SGD(model.parameters(),
                          lr=base_lr,
                          momentum=momentum,
                          weight_decay=weight_decay
                          )

    # 训练轮数
    start_epoch = 0
    max_epoch = train_cfg['max_epoch']
    epoch_size = len(dataset) // batch_size

    # 余弦退火学习率调度
    cos = True

    # 评估的轮数
    eval_epoch = 10

    # 开始训练循环
    print("---------------------------------------------------------")
    print("开始训练")
    t0 = time.time()
    for epoch in range(start_epoch, max_epoch):
        if cos and epoch > 20 and epoch <= max_epoch - 20:
            # use cos lr
            tmp_lr = 0.00001 + 0.5 * (base_lr - 0.00001) * (
                        1 + math.cos(math.pi * (epoch - 20) * 1. / (max_epoch - 20)))
            set_lr(optimizer, tmp_lr)

        elif cos and epoch > max_epoch - 20:
            tmp_lr = 0.00001
            set_lr(optimizer, tmp_lr)

        # use step lr
        else:
            if epoch in train_cfg['lr_epoch']:
                tmp_lr = tmp_lr * 0.1
                set_lr(optimizer, tmp_lr)

        for iter_i, (images, targets) in enumerate(dataloader):
            # 热身策略
            # if not args.no_warm_up:
            #     if epoch < args.wp_epoch:
            #         tmp_lr = base_lr * pow((iter_i + epoch * epoch_size) * 1. / (args.wp_epoch * epoch_size), 4)
            #         # tmp_lr = 1e-6 + (base_lr-1e-6) * (iter_i+epoch*epoch_size) / (epoch_size * (args.wp_epoch))
            #         set_lr(optimizer, tmp_lr)
            #
            #     elif epoch == args.wp_epoch and iter_i == 0:
            #         tmp_lr = base_lr
            #         set_lr(optimizer, tmp_lr)
            # to device
            images = images.to(device)

            # multi-scale trick
            if iter_i % 10 == 0 and iter_i > 0 and multi_scale:
                # randomly choose a new size
                size = random.randint(10, 19) * 32
                train_size = [size, size]
                model.set_grid(train_size)

            if multi_scale:
                # interpolate
                images = torch.nn.functional.interpolate(images, size=train_size, mode='bilinear', align_corners=False)

            # make train label
            targets = [label.tolist() for label in targets]
            targets = Tools.gt_creator(input_size=train_size, stride=yolo_net.stride, label_lists=targets)
            targets = torch.tensor(targets).float().to(device)

            # forward and loss
            conf_loss, cls_loss, txtytwth_loss, total_loss = model(images, target=targets)

            # backprop
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # display
            if iter_i % 10 == 0:
                if tensorboard:
                    # viz loss
                    writer.add_scalar('object loss', conf_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('class loss', cls_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('local loss', txtytwth_loss.item(), iter_i + epoch * epoch_size)

                t1 = time.time()
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                      '[Loss: obj %.2f || cls %.2f || bbox %.2f || total %.2f || size %d || time: %.2f]'
                      % (epoch + 1, max_epoch, iter_i, epoch_size, tmp_lr,
                         conf_loss.item(), cls_loss.item(), txtytwth_loss.item(), total_loss.item(), train_size[0],
                         t1 - t0),
                      flush=True)

                t0 = time.time()

        # evaluation
        # if (epoch + 1) % eval_epoch == 0:
        #     model.trainable = False
        #     model.set_grid(val_size)
        #     model.eval()
        #
        #     # evaluate
        #     # evaluator.evaluate(model)
        #
        #     # convert to training mode.
        #     model.trainable = True
        #     model.set_grid(train_size)
        #     model.train()

        # save model
        if (epoch + 1) % 10 == 0:
            print('Saving state, epoch:', epoch + 1)
            torch.save(model.state_dict(), os.path.join(path_to_save,
                                                        'model_' + repr(epoch + 1) + '.pth')
                       )


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
