#!/usr/bin/env Python
# coding=utf-8
import numpy as np
from torchvision import transforms  # 仅仅用来处理数据
import matplotlib.pyplot as plt
import sklearn.datasets as ds
from tqdm import tqdm
import os
import zipfile
import random
import json
import sys
from PIL import Image
from Btree_hw import btree

# 决策树实现图片分类, 不实现卷积, 只实现反向传播算法

# 绘画类
class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: self.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def set_axes(self, axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """Set the axes for matplotlib.
        Defined in :numref:`sec_calculus`"""
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        axes.grid()

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()


'''
参数配置
'''
train_parameters = {
    "input_size": [3, 64, 64],                                # 输入图片的shape
    "class_dim": -1,                                          # 分类数
    "target_path":"../resource/foods/",                     # 要解压的路径
    "train_list_path": "../resource/foods/train.txt",       # train.txt路径
    "eval_list_path": "../resource/foods/eval.txt",         # eval.txt路径
    "readme_path": "../resource/foods/readme.json",         # readme.json路径
    "label_dict":{},                                          # 标签字典
    "num_epochs": 2,                                          # 训练轮数
    "train_batch_size": 64,                                   # 训练时每个批次的大小
    "learning_strategy": {                                    # 优化函数相关的配置
        "lr": 0.01                                          # 超参数学习率
    }
}


# 生成数据列表
def get_data_list(target_path, train_list_path, eval_list_path):
    # 存放所有类别的信息
    class_detail = []
    # 获取所有类别保存的文件夹名称
    data_list_path = target_path + "foods/"
    class_dirs = os.listdir(data_list_path)
    # 总的图像数量
    all_class_images = 0
    # 存放类别标签
    class_label = 0
    # 存放类别数目
    class_dim = 0
    # 存储要写进eval.txt和train.txt中的内容
    trainer_list = []
    eval_list = []
    # 读取每个类别
    for class_dir in class_dirs:
        if class_dir != ".DS_Store":
            class_dim += 1
            # 每个类别的信息
            class_detail_list = {}
            eval_sum = 0
            trainer_sum = 0
            # 统计每个类别有多少张图片
            class_sum = 0
            # 获取类别路径
            path = data_list_path + class_dir
            # 获取所有图片
            img_paths = os.listdir(path)
            for img_path in img_paths:  # 遍历文件夹下的每个图片
                name_path = path + '/' + img_path  # 每张图片的路径
                if class_sum % 8 == 0:  # 每8张图片取一个做验证数据
                    eval_sum += 1  # test_sum为测试数据的数目
                    eval_list.append(name_path + "\t%d" % class_label + "\n")
                else:
                    trainer_sum += 1
                    trainer_list.append(name_path + "\t%d" % class_label + "\n")  # trainer_sum测试数据的数目
                class_sum += 1  # 每类图片的数目
                all_class_images += 1  # 所有类图片的数目

            # 说明的json文件的class_detail数据
            class_detail_list['class_name'] = class_dir  # 类别名称
            class_detail_list['class_label'] = class_label  # 类别标签
            class_detail_list['class_eval_images'] = eval_sum  # 该类数据的测试集数目
            class_detail_list['class_trainer_images'] = trainer_sum  # 该类数据的训练集数目
            class_detail.append(class_detail_list)
            # 初始化标签列表
            train_parameters['label_dict'][str(class_label)] = class_dir
            class_label += 1

            # 初始化分类数
    train_parameters['class_dim'] = class_dim

    # 乱序
    random.shuffle(eval_list)
    with open(eval_list_path, 'a') as f:
        for eval_image in eval_list:
            f.write(eval_image)

    random.shuffle(trainer_list)
    with open(train_list_path, 'a') as f2:
        for train_image in trainer_list:
            f2.write(train_image)

            # 说明的json文件信息
    readjson = {}
    readjson['all_class_name'] = data_list_path  # 文件父目录
    readjson['all_class_images'] = all_class_images
    readjson['class_detail'] = class_detail
    jsons = json.dumps(readjson, sort_keys=True, indent=4, separators=(',', ': '))
    with open(train_parameters['readme_path'], 'w') as f:
        f.write(jsons)
    print('生成数据列表完成！')


'''
参数初始化
'''
target_path = train_parameters['target_path']
train_list_path = train_parameters['train_list_path']
eval_list_path = train_parameters['eval_list_path']
batch_size = train_parameters['train_batch_size']

'''
解压原始数据到指定路径
'''

'''
划分训练集与验证集，乱序，生成数据列表
'''
# 每次生成数据列表前，首先清空train.txt和eval.txt
with open(train_list_path, 'w') as f:
    f.seek(0)
    f.truncate()
with open(eval_list_path, 'w') as f:
    f.seek(0)
    f.truncate()

# 生成数据列表
print("正在生成数据列表.......")
get_data_list(target_path, train_list_path, eval_list_path)


def pic_transform(image):
    # 定义转换操作
    resize = transforms.Resize((64, 64))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 依次应用转换
    image = resize(image)
    image = to_tensor(image)
    image = normalize(image)

    return image


class FoodDataset:
    """
    5类food数据集类的定义
    """
    def __init__(self, mode='train'):
        """
        初始化函数
        """
        self.data = []

        with open(target_path + '{}.txt'.format(mode)) as f:
            for line in f.readlines():
                info = line.strip().split('\t')
                if len(info) > 0:
                    self.data.append([info[0].strip(), info[1].strip()])

        self.transforms = pic_transform

    def __getitem__(self, index):
        """
        根据索引获取单个样本
        """
        image_file, label = self.data[index]
        image = Image.open(image_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.transforms(image)
        return image.numpy(), np.array(label, dtype='int64')

    def __len__(self):
        """
        获取样本总数
        """
        return len(self.data)


'''
构造数据提供器
'''
print("加载训练集.....")
train_dataset = FoodDataset(mode='train')
print("加载验证集.....")
eval_dataset = FoodDataset(mode='eval')


print("训练集长度: ", train_dataset.__len__())
print("验证集长度: ", eval_dataset.__len__())

# 训练决策树，采用4层的ID3树
bt1 = btree(method='ID3', depth=4)

def train(train_dataset, eval_dataset):
    bt1.fit_data(train_dataset)
    # 获取决策树的分叉详情
    # t_dict = bt1.tree
    # 查看最佳分裂点
    # node_list1 = bt1.node_list
    # 查看特征重要度排序, 返回的是变量的index，越靠前的变量越重要。
    # feature_importance = bt1.feature_importance


def predict(eval_dataset):
    def compare_result(predict, test):
        count = 0
        for i, j in zip(predict.tolist(), test.tolist()):
            if i == j:
                count += 1
        return count / len(predict)

    eval_label = eval_dataset.data
    y_predict = bt1.predict(test_data)
    return compare_result(y_predict, test_label)


if __name__ == "__main__":
    # 训练
    train(train_dataset, eval_dataset)

    # 预测
    # print("result: ", predict(eval_dataset))

    # 样本映射
    LABEL_MAP = ['beef_tartare', 'baklava', 'beef_carpaccio', 'apple_pie', 'baby_back_ribs']
    # In[18]
    # # 随机取样本展示
    # indexs = [2, 38, 56, 92, 100, 101]
    #
    # for idx in indexs:
    #     predict_label = np.argmax(result[0][idx])
    #     real_label = eval_dataset.__getitem__(idx)[1]
    #     print('样本ID：{}, 真实标签：{}, 预测值：{}'.format(idx, LABEL_MAP[real_label], LABEL_MAP[predict_label]))