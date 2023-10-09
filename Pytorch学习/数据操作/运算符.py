import torch

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y, x - y, x * y, x / y, x ** y)  # **运算符是求幂运算
print(torch.exp(x))

X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# 连结操作
print(torch.cat((X, Y), dim=0))
print(torch.cat((X, Y), dim=1))

# 判断二元张量的每个位置是否相同
print(X == Y)

print(X.sum())


