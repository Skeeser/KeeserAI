#!/usr/bin/env Python
# coding=utf-8
import numpy as np

class Tree():
    def __init__(self, node, impurity, depth, left, right, is_leaf, label, index, split):
        self.tree = {"num": node, "impurity": impurity, "depth": depth, "left": left, "right": right,
                     "is_leaf": is_leaf, "label": label, "index": index, "split": split}

    def generate_tree_path(self, path):
        dict_index = ""
        for i in path:
            if i == "0":
                dict_index = dict_index + "0"
            else:
                last_l = dict_index.rfind("0")
                dict_index = dict_index[:last_l] + "1"
        return dict_index.replace("1", '["right"]').replace("0", '["left"]')

    def add_node(self, path, node, impurity, depth, left, right, is_leaf, label, index, split):
        tree_path_index = self.generate_tree_path(path)
        set_dict = {"num": node, "impurity": impurity, "depth": depth, "left": left, "right": right, "is_leaf": is_leaf,
                    "label": label, "index": index, "split": split}
        exec("self.tree" + tree_path_index.__str__() + " = set_dict")


class btree():
    def __init__(self, method='ID3', sample_weight=None, depth=10, min_impurity=0, min_samples_split=2):
        self.method = method
        self.sample_weight = []
        self.node_list = []
        self.feature_importance = []
        self.depth = depth
        self.min_impurity = min_impurity
        self.min_samples_split = min_samples_split
        self.t = None
        self.path = ''

    def group_count(self, array):
        groups = np.unique(array)
        count_array = np.array(list(map(lambda x: array[array == x].__len__(), groups)))
        return count_array.astype(int)

    def calc_entropy(self, label):
        label_count = self.group_count(label)
        return sum(-label_count / label_count.sum() * np.log2(label_count / label_count.sum()))

    def calc_gini(self, label):
        label_count = self.group_count(label)
        return 1 - sum((label_count / label_count.sum()) ** 2)

    def calc_impurity(self, combine, left_combine, right_combine):
        total_entropy = self.calc_entropy(combine[:, -1])
        if self.method != 'CART':
            entropy_left = self.calc_entropy(left_combine[:, -1])
            entropy_right = self.calc_entropy(right_combine[:, -1])
            entropy_node = left_combine.shape[0] / combine.shape[0] * entropy_left + right_combine.shape[0] / \
                           combine.shape[0] * entropy_right
            if self.method == 'ID3':
                entropy_increment = total_entropy - entropy_node
                impurity = entropy_increment

            elif self.method == 'C45':
                entropy_increment = total_entropy - entropy_node
                split_node = np.hstack((left_combine[:, -1], right_combine[:, -1]))
                entropy_split_node = self.calc_entropy(split_node)
                entropy_ratio = entropy_increment / entropy_split_node
                impurity = entropy_ratio
        else:
            gini_left = self.calc_gini(left_combine[:, -1])
            gini_right = self.calc_gini(right_combine[:, -1])
            gini = left_combine.shape[0] / combine.shape[0] * gini_left + right_combine.shape[0] / combine.shape[
                0] * gini_right
            impurity = 1 - gini
        return impurity

    # 每个节点遍历计算
    def continuous_variable_node(self, combine):
        sorted_data = np.sort(combine[:, :-1], axis=0)
        sorted_list = sorted_data.T.tolist()
        best_node_list = list()

        for index_ in range(len(sorted_list)):
            sorted_ = sorted_list[index_]
            sorted_set = sorted(list(set(sorted_)))
            max_impurity = -np.Inf

            node_l = [(sorted_set[i] + sorted_set[i + 1]) / 2 for i in range(len(sorted_set)) if
                      i <= len(sorted_set) - 2]
            for node_ in node_l:
                left_combine = combine[np.where(combine[:, index_] <= node_)[0], :]
                right_combine = combine[np.where(combine[:, index_] > node_)[0], :]
                impurity = self.calc_impurity(combine, left_combine, right_combine)
                if impurity >= max_impurity:
                    max_impurity = impurity
                    best_node = node_

            # print(self.method + ' max value: ' + str(max_impurity))
            # print(combine_df[combine_df[index_] <= best_node].groupby(combine_df['label'])['label'].count())
            # print(combine_df[combine_df[index_] > best_node].groupby(combine_df['label'])['label'].count())
            best_node_list.append(best_node)
        return best_node_list

    def get_feature_importance_index(self, combine):
        impurity_list = list()
        for index_, node_ in enumerate(self.node_list):
            left_combine = combine[np.where(combine[:, index_] <= node_)[0], :]
            right_combine = combine[np.where(combine[:, index_] > node_)[0], :]
            impurity = self.calc_impurity(combine, left_combine, right_combine)
            impurity_list.append(impurity)
        sorted_index = sorted(range(len(impurity_list)), key=lambda k: impurity_list[k], reverse=True)
        sorted_index_filtered = [i for i in sorted_index if i not in self.feature_importance]
        if len(sorted_index_filtered) != 0: return sorted_index_filtered[0]

    def cbind(self, data, label):
        combine = np.column_stack((data, label))
        return combine

    #   #每一层的节点选择为同一特征
    def tree_growth(self, combine, depth):
        # selected_index = self.sorted_feature_importance[depth]
        self.feature_importance.append(self.get_feature_importance_index(combine))
        selected_index = self.feature_importance[depth]
        left_combine = combine[np.where(combine[:, selected_index] <= self.node_list[selected_index])[0], :]
        right_combine = combine[np.where(combine[:, selected_index] > self.node_list[selected_index])[0], :]
        return [left_combine, right_combine]

    def build_tree(self, combine, depth=0):
        child_df = self.tree_growth(combine, depth)
        left, right = child_df[0], child_df[1]

        impurity = self.calc_impurity(combine, left, right)

        # start growth if satisfies the conditions
        if impurity > self.min_impurity and depth < self.depth and left.shape[0] >= self.min_samples_split and \
                right.shape[0] >= self.min_samples_split:
            if depth == 0:
                # initialize Tree class
                self.t = Tree({"total": combine.shape[0],
                               "group count": dict(zip(np.unique(combine[:, -1]), self.group_count(combine[:, -1])))},
                              impurity,
                              0,
                              dict(zip(np.unique(left[:, -1]), self.group_count(left[:, -1]))),
                              dict(zip(np.unique(right[:, -1]), self.group_count(right[:, -1]))),
                              False,
                              np.unique(combine[:, -1])[0] if len(
                                  self.group_count(combine[:, -1])) == 1 else self.group_count(combine[:, -1]).argmax(),
                              self.feature_importance[depth],
                              self.node_list[self.feature_importance[depth]])
            else:
                # add node for tree dict.
                self.t.add_node(self.path,
                                {"total": combine.shape[0],
                                 "group count": dict(zip(np.unique(combine[:, -1]), self.group_count(combine[:, -1])))},
                                impurity,
                                depth,
                                dict(zip(np.unique(left[:, -1]), self.group_count(left[:, -1]))),
                                dict(zip(np.unique(right[:, -1]), self.group_count(right[:, -1]))),
                                False,
                                np.unique(combine[:, -1])[0] if len(
                                    self.group_count(combine[:, -1])) == 1 else self.group_count(
                                    combine[:, -1]).argmax(),
                                self.feature_importance[depth],
                                self.node_list[self.feature_importance[depth]])
            # growing tree by left child node and right child node consecutively in a for loop.
            for i in range(len(child_df)):
                self.path = self.path + str(i)
                df = child_df[i]
                # here starts the recurse by input dataset and depth
                self.build_tree(df, depth + 1)
        # else add leaf node, where the parameter "is_leaf" would be True.
        else:
            self.t.add_node(self.path,
                            {"total": combine.shape[0],
                             "group count": dict(zip(np.unique(combine[:, -1]), self.group_count(combine[:, -1])))},
                            impurity,
                            depth,
                            dict(zip(np.unique(left[:, -1]), self.group_count(left[:, -1]))),
                            dict(zip(np.unique(right[:, -1]), self.group_count(right[:, -1]))),
                            True,
                            np.unique(combine[:, -1])[0] if len(
                                self.group_count(combine[:, -1])) == 1 else self.group_count(combine[:, -1]).argmax(),
                            self.feature_importance[depth],
                            self.node_list[self.feature_importance[depth]])

    def fit(self, data, label):
        combine = self.cbind(data, label)
        self.node_list = self.continuous_variable_node(combine)
        self.build_tree(combine)

    def fit_data(self, dataset):
        self.node_list = self.continuous_variable_node(dataset)
        self.build_tree(dataset)

    def predict_main(self, data):
        dic = self.t.tree
        while dic['is_leaf'] == False:
            if data[dic['index']] <= dic['split']:
                dic = dic['left']
            else:
                dic = dic['right']
        label = dic['label']
        return label

    def predict(self, data):
        return np.apply_along_axis(self.predict_main, 1, data)

    def score(self, predict, test):
        count = 0
        for i, j in zip(predict.tolist(), test.tolist()):
            if i == j:
                count += 1
        return count / len(predict)

    @property
    def tree(self):
        return self.t.tree
