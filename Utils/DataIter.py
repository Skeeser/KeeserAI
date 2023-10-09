import random
import torch


# 接收批量大小、特征矩阵和标签向量作为输入，
# 生成大小为batch_size的小批量。 每个小批量包含一组特征和标签。
class DataIter:
    @staticmethod
    def data_iter(batch_size, features, labels):
        num_examples = len(features)
        indices = list(range(num_examples))
        # 这些样本是随机读取的，没有特定的顺序
        random.shuffle(indices)
        for i in range(0, num_examples, batch_size):
            batch_indices = torch.tensor(
                indices[i: min(i + batch_size, num_examples)])
            yield features[batch_indices], labels[batch_indices]


if __name__ == "__main__":
    batch_size = 10

    # for X, y in DataIter.data_iter(batch_size, features, labels):
    #     print(X, '\n', y)
    #     break