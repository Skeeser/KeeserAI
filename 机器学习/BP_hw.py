import numpy as np
from sklearn import model_selection  # 仅仅用来划分数据集
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


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


# 将字符串转为整型，便于数据加载
def iris_type(s):
    it = {b'Iris-setosa':0, b'Iris-versicolor':1, b'Iris-virginica':2}
    return it[s]


# 加载数据
data_path='../resource/iris/iris.data'  # 数据文件的路径
data = np.loadtxt(data_path,               # 数据文件路径
                  dtype=float,              # 数据类型
                  delimiter=',',            # 数据分隔符
                  converters={4:iris_type})  # 将第5列使用函数iris_type进行转换
# 数据分割
x, y = np.split(data,                      # 要切分的数组
                (4,),                      # 沿轴切分的位置，第5列开始往后为y
                axis=1)                    # 代表纵向分割，按列分割

# print(x)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,              # 所要划分的样本特征集
                                                               y,               # 所要划分的样本结果
                                                               random_state=1,  # 随机数种子
                                                               test_size=0.2)   # 测试样本占比
# 按列拼接两个数组
train_data = np.concatenate([x_train, y_train], axis=1)
test_data = np.concatenate([x_test, y_test], axis=1)


# 定义激活函数（Sigmoid）
def sigmoid(in_x):
    return 1 / (1 + np.exp(-in_x))

class BPNet:
    def __init__(self):
        # 超参数
        input_size = 4
        hidden_size = 8
        output_size = 3
        self.L2_param = 0.001

        TRAIN_FLAG = True
        self.W1 = np.random.normal(0.0, input_size**-0.5, (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.normal(0.0, hidden_size**-0.5, (hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

        if not TRAIN_FLAG and os.path.exists('.saved_weights.npz'):
            loaded_weights = np.load('.saved_weights.npz')
            self.W1 = loaded_weights['W1']
            self.b1 = loaded_weights['B1']
            self.W2 = loaded_weights['W2']
            self.b2 = loaded_weights['B2']

    # 训练模型
    def train(self, train_data, test_data, learning_rate=1e-4):
        learning_rate = learning_rate
        last_test_loss = 99999.0
        flag_num = 0
        num_epochs = 100000
        animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                                    legend=['train loss', 'test loss'])

        for epoch in tqdm(range(num_epochs), desc='Training'):
            for X1, X2, X3, X4, y in train_data:
                Y = [0, 0, 0]
                Y[int(y)] = 1
                X = np.array([[X1, X2, X3, X4]])
                # 前向传播
                z1 = X@self.W1 + self.b1
                a1 = sigmoid(z1)
                z2 = a1@self.W2 + self.b2
                a2 = sigmoid(z2)
                # 计算损失（均方误差）
                train_loss = np.mean(np.square(Y - a2))
                # 反向传播
                d2 = (a2 - Y) * a2 * (1 - a2)
                d1 = d2@self.W2.T * a1 * (1 - a1)
                # 更新权重和偏差
                self.W2 -= learning_rate * (np.dot(a1.T, d2) + self.L2_param * self.W2)
                self.b2 -= learning_rate * np.sum(d2, axis=0, keepdims=True)
                self.W1 -= learning_rate * (np.dot(X.T, d1) + self.L2_param * self.W1)
                self.b1 -= learning_rate * np.sum(d1, axis=0, keepdims=True)

            for X1, X2, X3, X4, y in test_data:
                Y = [0, 0, 0]
                Y[int(y)] = 1
                X = np.array([[X1, X2, X3, X4]])
                # 前向传播
                z1 = X@self.W1 + self.b1
                a1 = sigmoid(z1)
                z2 = a1@self.W2 + self.b2
                a2 = sigmoid(z2)
                index = np.argmax(a2)
                # 计算损失（均方误差）
                test_loss = np.mean(np.square(Y - a2))

            # 如果test_loss不再降低, 降低学习率
            # if last_test_loss < test_loss:
            #     flag_num += 1
            # else:
            #     flag_num = 0
            #
            # if flag_num == 20:
            #     flag_num = 0
            #     learning_rate *= 0.1
            #     print(f"learning rate down, is {learning_rate}")
            #
            # last_test_loss = test_loss

            if (epoch + 1) % 100 == 0:
                animator.add(epoch + 1, (train_loss, test_loss))
            if (epoch + 1) % 1000 == 0:
                print(f"[epoch: {epoch + 1} || train_loss: {train_loss} || test_loss: {test_loss}]")

        np.savez('./saved_weights.npz', W1=self.W1, B1=self.b1, W2=self.W2, B2=self.b2)
        plt.show()

    # 测试模型
    def accuracy(self, test_data):
        z1 = np.dot(test_data, self.W1) + self.b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = sigmoid(z2)
        index = np.argmax(a2)

        for i in range(len(test_data)):
            ret = round(a2[i][0])
            print(f"Input: {test_data[i]}, Output: {ret}")


bp = BPNet()
# BP网络训练
bp.train(train_data, test_data)
