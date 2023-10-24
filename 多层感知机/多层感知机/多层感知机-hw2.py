import numpy as np

# 定义数据集
data = np.array([[[0, 0], [0]],
                 [[0, 1], [1]],
                 [[1, 0], [1]],
                 [[1, 1], [0]]])


# 定义激活函数（Sigmoid）
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 初始化网络参数
input_size = 2
hidden_size = 4
output_size = 1
learning_rate = 0.01

W1 = np.random.rand(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.rand(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# 训练模型
for epoch in range(10000):
    for X, y in data:
        X = np.array([X])
        # 前向传播
        z1 = X@W1 + b1
        a1 = sigmoid(z1)
        z2 = a1@W2 + b2
        a2 = sigmoid(z2)
        # 计算损失（均方误差）
        loss = np.mean(np.square(y - a2))

        # 反向传播
        d2 = a2 - y
        d1 = d2@W2.T * a1 * (1 - a1)

        # 更新权重和偏差
        W2 -= learning_rate * np.dot(a1.T, d2)
        b2 -= learning_rate * np.sum(d2, axis=0, keepdims=True)
        W1 -= learning_rate * np.dot(X.T, d1)
        b1 -= learning_rate * np.sum(d1, axis=0, keepdims=True)

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

# 测试模型
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
z1 = np.dot(test_data, W1) + b1
a1 = sigmoid(z1)
z2 = np.dot(a1, W2) + b2
a2 = sigmoid(z2)

print("Test Results:")
for i in range(len(test_data)):
    ret = round(a2[i][0])
    print(f"Input: {test_data[i]}, Output: {ret}")
