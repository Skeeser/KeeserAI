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


# 自定义分割函数,为了处理pytorch分割字符串时无法指定去掉标点符号问题
def split_(text):
    # 先把大写字母全部小写
    text = text.lower()
    # 定义要过滤的标点符号集合
    delete_use = "!\"#$%&()*+.,-/:;=?@[\]^_`{|}~"
    # 全部用空格代替
    for i in delete_use:
        text= text.replace(i,' ')
    # 分割字符
    tokens = text.split()
    # 返回结果
    return tokens


# 文本预处理，通过词向量获得分词并构建词汇表
def get_token(train_caption):
    # 定义分割器
    tokenizer = torchtext.data.utils.get_tokenizer(split_, language='en')
    # 多套一层<start>和<end>，用于后续输出可以有<start>和<end>
    train_caption = ['<start> ' + caption + ' <end>' for caption in train_caption]
    # 创建词汇表,根据给出的训练集合寻找内容,在词典之外的词语用<unknown_word>代替,只保留出现过的词语
    # 最大词汇表容量为5000
    tokenizer = torchtext.vocab.build_vocab_from_iterator(map(tokenizer, train_caption),
                                        specials=['<unknown_word>'], min_freq=1,
                                        max_tokens=top_k)
    # 返回词汇表
    return tokenizer


# 根据词典进行文本-索引的序列化的自定义函数
def Word2Sequence(tokenizer, train_captions):
    # 序列化结果
    train_seqs = []
    # 遍历每句话
    for train_caption in train_captions:
        # 对句子分词
        tokens = split_(train_caption)
        # 临时存储变量
        temp = []
        # 遍历句子的每个分词
        for token in tokens:
            # 在词典中就记录索引值,unknown_word此时被记录为了1
            if token in tokenizer:
                temp.append(tokenizer[token])
            # 不在则记录0
            else:
                temp.append(PAD)
        # 记录序列化结果
        train_seqs.append(temp)
    return train_seqs


# captions的预处理全过程
def captions_preprocess(train_captions):
    # 根据给出的train_captions获得词汇表
    tokenizer = get_token(train_captions)
    # 进行序列化
    cap_vector = Word2Sequence(tokenizer, train_captions)
    return tokenizer, cap_vector


# 自定义数据集的读取和加载,继承自Dataset
class CustomDataset(Dataset):
    # 将读取的文件名称和captions序列存储,transform设置为None
    def __init__(self, img_names, captions, transform=None):
        self.img_names = img_names
        self.captions = captions
        self.transform = transform

    # 数据集长度定义为图片名数量
    def __len__(self):
        return len(self.img_names)

    # 读取数据集的返回item
    def __getitem__(self, idx):
        # 从给定索引获得captions的对应序列
        cap = self.captions[idx]
        # 从给定索引获得对应图片名
        img_name = self.img_names[idx]
        # 图片名转为tensor(244,244)的形式,用之前的load_image完成
        img_tensor, img_name = load_image(img_name)
        cap = torch.Tensor(cap)
        return img_tensor, cap, img_name


# 手动将抽取出的批处理样本进行堆叠返回结果
def collate_fn(data):
    # 对CustomDataset返回的样本进行排序,排序依据为cpations,从大到小排序
    data.sort(key=lambda x: len(x[1]), reverse=True)
    # 获得批处理样本并读取结果
    images, captions, img_names = zip(*data)
    # 合并images和img_names,从元组3维tensor转为4维tensor
    images = torch.stack(images, 0)
    # 合并captions从元组1维tensor转为2维tensor
    # lengths记录captions的每个元素长度,用于后续压缩
    lengths = [len(cap) for cap in captions]
    # 判断句子的最大长度是否超过了MAX_LEN
    len_use = 0
    if max(lengths) > MAX_LEN:
        # 超过了就截断部分句子，补全部分句子
        len_use = MAX_LEN
    else:
        # 没超过全部补全
        len_use = max(lengths)
    # 按最大长度初始化新的captions,由于PAD为0,此处直接用zeros填充内容
    new_captions = torch.zeros(len(captions), len_use).long()
    # 遍历captions
    for i, cap in enumerate(captions):
        # 读取词语长度，如果超过了MAX_LEN补到MAX_LEN
        if lengths[i] <= MAX_LEN:
            end = lengths[i]
        else:
            # 注意lengths[i]也不能超过MAX_LEN
            lengths[i] = MAX_LEN
            end = MAX_LEN
        # 将captions对应内容补充到new_captions内
        new_captions[i, :end] = cap[:end]
    # 返回结果
    return images, new_captions, lengths, img_names


# 自定义获得数据集的函数，输入为caption文本和图片名集合
def get_ds(Cap, img, batch):
    # captions的预处理
    tokenizer, cap = captions_preprocess(Cap)
    # 将数据集分割为训练数据集和测试数据集,数量比值为4:1
    img_name_train, img_name_val, cap_train, cap_val = train_test_split(
        img, cap, test_size=0.2, random_state=0)
    # 获得训练数据集、加载器和测试数据集、加载器
    # 注意不打乱顺序(前面已打乱过,训练集batch_size大小为16,太大会导致内存不够用，测试机为训练集batch_size大小的四分之一)
    train_dataset = CustomDataset(img_name_train, cap_train)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=False, collate_fn=collate_fn)
    print(len(train_dataset))
    val_dataset = CustomDataset(img_name_val, cap_val)
    val_loader = DataLoader(val_dataset, batch_size=int(batch / 4), shuffle=False, collate_fn=collate_fn)
    # 返回结果
    return tokenizer, train_loader, val_loader, img_name_val, cap_val, img_name_train, cap_train


# 取data_size条数据集信息
train_captions, img_name_vector = img_cap_list(Size=data_size)
tokenizer, train_loader, val_loader, img_name_val, cap_val, img_name_train, cap_train = get_ds(train_captions,img_name_vector,batch)


# 实现看图说话模型
class ImageCaption(nn.Module):
    def __init__(self):
        super(ImageCaption, self).__init__()
        # 编码器是CNN网络
        self.encoder = nn.Sequential(models.resnet152(),
                                     nn.Linear(2048, img_ev_size))

        # 解码器是LSTM网络
        self.decoder = None


    def encoder_func(self, x):
        pass

    def decoder_func(self, x):
        pass


if __name__ == "__main__":
    # 定义批处理大小
    batch = 16
    # 定义最大词汇表容量
    top_k = 5000
    # 定义句子的最大长度
    MAX_LEN = 30
    # 定义自动填充内容
    PAD = 0
    # 定义要处理的文本数量
    data_size = 10000
    # 定义图片特征向量
    img_ev_size = 256

