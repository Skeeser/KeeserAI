import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

# input_data = torch.tensor(input_data)
# output_data = torch.tensor(output_data)
# xtrain = torch.unsqueeze(torch.tensor(xtrain).float(), dim=1)   #升维
# xtest = torch.unsqueeze(torch.tensor(xtest).float(), dim=1)
# ytrain = ytrain + 0.2*torch.rand(ytrain.size())     #加入噪声提高鲁棒性
# ytest = ytest + 0.2*torch.rand(ytest.size())


class BPNet:
    def __init__(self):
        self.net = nn.Sequential(
            torch.nn.Linear(2, 12),
            torch.nn.ReLU(),  # 激活函数
            torch.nn.Linear(12, 1),
            torch.nn.Softplus()
        )

    def run(self):
        pass


if __name__ == "__main__":
    bp = BPNet()
    bp.run()