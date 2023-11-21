import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import json
from sklearn.utils import shuffle
from PIL import Image, ImageOps


# caption文件预处理,默认输入Size为-1
def img_cap_list(Size: int = -1):
    # 图像路径和情景识别json文件路径
    image_path = "../resource/coco\\train2014\\"
    cap_train_json_path = "../resource/coco\\annotations\\captions_train2014.json"
    # 打开情景识别文件并加载
    with open(cap_train_json_path, 'r') as f:
        annotations = json.load(f)
    # 对图像的文件名和其caption文本建立列表
    all_captions = []
    all_img_name_vector = []
    # 从情景识别文件数据中遍历annotations数组的每个元素
    # 元素包含三个标签，分别是image_id,id,caption,对应图像id,id类型编号和文本描述
    for annot in annotations['annotations']:
        # 从元素中提取caption标签内容,前后加上<start>和<end>
        caption = '<start> ' + annot['caption'] + ' <end>'
        # 提取image_id标签内容,对应id值
        image_id = annot['image_id']
        # 从image_id转为图像文件名称
        #  '%012d'对应000000000000,取余image_id即image_id补齐12位0
        full_coco_image_path = image_path + 'COCO_train2014_' + '%012d.jpg' % (image_id)
        # 将文件名和caption文本添加到对应列表中
        all_img_name_vector.append(full_coco_image_path)
        all_captions.append(caption)
    # 同时打乱文本和图片顺序
    train_captions, img_name_vector = shuffle(all_captions, all_img_name_vector, random_state=1)
    # total 414113,注意此时一个图像对应的文本可能不唯一,但按照文本计算数量
    # Size为-1提取全部信息
    if Size == -1:
        train_captions = train_captions[:]
        img_name_vector = img_name_vector[:]
    # 否则提取Size数量的图片名和caption文本
    else:
        train_captions = train_captions[:Size]
        img_name_vector = img_name_vector[:Size]
    return train_captions, img_name_vector


# pytorch版本加载图片函数
def load_image(image_path):
    # 定义图片的transform
    # 为了适应ResNet网络,图片大小为244*244,并转为张量形式
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # 通过RGB格式加载图片
    image = Image.open(image_path).convert('RGB')
    # 图片类型转换
    image = transform(image)
    # 返回图片和图片路径
    return image, image_path


# 不修改图片版本
def load_image_v2(image_path):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image, image_path