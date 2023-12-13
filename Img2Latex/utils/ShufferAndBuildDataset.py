import random
import os

ROOT_PATH = r"../../resource/yolo/"

# 读取文件中的 ID 列表
with open(ROOT_PATH + 'yolo_ids.txt', 'r') as file:
    ids = file.readlines()

# 去除每行末尾的换行符
ids = [id.strip() for id in ids]

# 设置划分比例，比如 80% 训练集，20% 验证集
train_ratio = 0.8
validation_ratio = 1 - train_ratio

# 计算划分数量
num_samples = len(ids)
num_train_samples = int(train_ratio * num_samples)
num_validation_samples = num_samples - num_train_samples

# 随机打乱 ID 列表
random.shuffle(ids)

# 根据比例划分训练集和验证集
train_ids = ids[:num_train_samples]
validation_ids = ids[num_train_samples:]

# 写入训练集和验证集的 ID 到对应文件
with open(ROOT_PATH + 'train_ids.txt', 'w') as train_file:
    for train_id in train_ids:
        train_file.write(train_id + '\n')

with open(ROOT_PATH + 'validation_ids.txt', 'w') as validation_file:
    for validation_id in validation_ids:
        validation_file.write(validation_id + '\n')