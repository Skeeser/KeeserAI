"""
有时，我们希望将某些计算移动到记录的计算图之外。
例如，假设y是作为x的函数计算的，而z则是作为y和x的函数计算的。
想象一下，我们想计算z关于x的梯度，但由于某种原因，希望将y视为一个常数， 并且只考虑到x在y被计算后发挥的作用。
"""
import torch

x = torch.arange(4.0)
print(x)

x.grad.zero_()
y = x * x
# 分离y来返回一个新变量u，该变量与y具有相同的值， 但丢弃计算图中如何计算y的任何信息。
# 将u当做常数处理
u = y.detach()
z = u * x

z.sum().backward()
print(x.grad == u)

# 由于记录了y的计算结果，我们可以随后在y上调用反向传播， 得到y=x*x关于的x的导数，即2*x。
x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)

