import math
import time
import numpy as np
import torch
from d2l import torch as d2l
from Utils.Timer import Timer

n = 10000

a = torch.ones([n])
b = torch.ones([n])
c = torch.zeros([n])

timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]

print(f'{timer.stop():.5f} sec')

timer.start()
d = a + b
print(f'{timer.stop():.5f} sec')