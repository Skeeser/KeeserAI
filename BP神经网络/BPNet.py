import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
from Animator import Animator
# from RobotCtrl import RobotCtrl
from TrainBotCtrl import RobotCtrl

FILE_PATH = "../model/pid.pth"
PRINT_FLAG = True  # print很占用性能
rc = RobotCtrl()

# 0.5以下不太收敛了


def desired_val(nums):
    step = 10.0  # 6.0
    set_val = 75.0
    desired_val = min(set_val, nums * step)
    # rc.set_speed(0.6, 0.6)
    return [desired_val, desired_val]


def run(uu):
    uu = uu.item()
    rc.set_uu(uu, uu)


class BPNet:
    def __init__(self, nh=8, xite=2, alfa=0.5, i=0):
        """
        0.6 (self, nh=8, xite=2, alfa=0.5, i=0):
        0.4 (self, nh=8, xite=2.5, alfa=0.85, i=0):
        本函数有四个外部输入变量：T，nh，xite,alfa
        T输入采样时间，nh确定隐含层层数，
        xite和alfa权值系数修正公式里的学习速率和惯性系数。
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"cuda可用,数量为:{torch.cuda.device_count()}")

        self.nh = nh
        self.xite = xite
        self.alfa = alfa
        self.i = i
        self.wi_2 = torch.normal(0, 0.1, size=(nh, 3)).to(self.device)
        self.wo_2 = torch.normal(0, 0.1, size=(3, nh)).to(self.device)
        self.wi_1 = torch.normal(0, 0.1, size=(nh, 3)).to(self.device)
        self.wo_1 = torch.normal(0, 0.1, size=(3, nh)).to(self.device)
        self.wo = torch.zeros(size=(3, nh)).to(self.device)
        self.wi = torch.zeros(size=(nh, 3)).to(self.device)

        self.err = 0.0
        self.err1 = 0.0
        self.err2 = 0.0
        self.y = 0.0
        self.y1 = 0.0
        # 期望输出
        self.r = 0.0
        self.uu = 0.0
        self.u1 = 0.0

    def Sigmoid(self, X):
        # return 2 / (torch.exp(X) + torch.exp(-X)).pow(2)  # Sigmoid激活函数
        return torch.sigmoid(X).to(self.device)

    def Tanh(self, X):
        # return torch.exp(X) / (torch.exp(X) + torch.exp(-X))  # tanh激活函数 不是
        return torch.tanh(X).to(self.device)

    def forward(self, num_epoch):
        self.r = desired_val(num_epoch)[self.i] / 100
        self.y = rc.get_speed()[self.i] / 100   # 放缩
        self.err = self.r - self.y
        constant = 0.5  # 防止超调参数  线性增长0.2
        if PRINT_FLAG:
            print(f"desired:{self.r}, getspeed:{self.y}, err:{self.err}")
        xi = torch.Tensor([[self.r, self.y, self.err]]).to(self.device)
        xx = torch.Tensor([[self.err - self.err1], [self.err],
                          [self.err + self.err2 - 2 * self.err1]]).to(self.device)

        I = torch.matmul(xi, self.wi_1.t())  # 隐藏层输入 线性，要不要加偏置？
        Oh = self.Tanh(I)
        O = torch.matmul(self.wo_1, Oh.t())  # 输出层输入
        K = self.Sigmoid(O)  # K=[Kp,Ki,Kd]，维数为1*3
        # print(K)
        # 这里防止超调, 只有一个判断
        if self.err > constant:  # pd控制
            K[1][0] = 0

        self.uu = self.u1 + torch.matmul(K.t(), xx)  # pid公式算出输出量

        dyu = torch.sign((self.y - self.y1) / (self.uu -
                         self.u1 + 0.0000001))  # 计算增量，变化趋势, 近似偏导
        dK = self.Sigmoid(K)  # 激活函数

        delta3 = self.err * dyu * xx * dK  # 修正函数
        self.wo = self.wo_1 + self.xite * \
            torch.matmul(delta3, Oh) + self.alfa * (self.wo_1 - self.wo_2)
        dOh = self.Sigmoid(Oh)  # 激活函数
        # 修正函数
        self.wi = self.wi_1 + self.xite * torch.matmul((dOh * torch.matmul(delta3.t(), self.wo)).t(), xi) \
            + self.alfa * (self.wi_1 - self.wi_2)
        self.update()  # 更新数据
        return self.uu, K.t().tolist()[0], self.r, self.y

    def update(self):
        self.err2 = self.err1
        self.err1 = self.err
        self.y1 = self.y
        self.u1 = self.uu
        self.wo_2 = self.wo_1
        self.wi_2 = self.wi_1
        self.wo_1 = self.wo
        self.wi_1 = self.wi

    def train(self, num_epochs):
        Kp, Ki, Kd = 0, 0, 0
        # 定义绘图
        animator = Animator(xlabel='epoch', legend=['Kp', 'Ki', 'Kd', 'getspeed'],
                            xlim=[1, num_epochs])
        for epoch in range(num_epochs):
            uu, K, disired, getspeed = self.forward(epoch)
            # print(K.tolist()[0])
            Kp, Ki, Kd = K
            if PRINT_FLAG:
                print(f"uu:{uu.item()}, K:{K}")
                animator.add(epoch + 1, K + [getspeed])
            run(uu)

        rc.stop()
        if PRINT_FLAG:
            print(f"Kp:{Kp} Ki:{Ki} Kd:{Kd}")

        # 打开一个文本文件，如果文件不存在则创建
        if self.i == 0:
            with open(FILE_PATH, 'w') as file:
                file.write(f"{Kp} {Ki} {Kd}\n")
        else:
            with open(FILE_PATH, 'a') as file:
                file.write(f"{Kp} {Ki} {Kd}\n")

        if PRINT_FLAG:
            plt.show()


if __name__ == "__main__":
    num_epochs = 100
    if len(sys.argv) == 1:
        print("传入参数过少, 加上--left / --right")
    elif len(sys.argv) == 2:
        if sys.argv[1] == "--left":
            left_bp = BPNet(i=0)
            print("当前训练左轮pid")
            left_bp.train(num_epochs)
        elif sys.argv[1] == "--right":
            right_bp = BPNet(i=1)
            print("当前训练右轮pid")
            right_bp.train(num_epochs)
        else:
            print("传入参数错误")
    else:
        print("传入参数过多, 发生错误")
