import torchtext.vocab as vocab

# 假设有一个文本数据迭代器 data_iter，其中每个元素是一个单词序列（如列表）
data_iter = [['this', 'is', 'a', 'sentence'], ['another', 'sentence']]

# 使用 build_vocab_from_iterator 构建词汇表
vocab.build_vocab_from_iterator(data_iter)