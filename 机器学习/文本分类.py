from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
import pandas as pd
import time


# 读取垃圾邮件数据
df = pd.read_csv('../resource/email/emails.csv')
X = df['text'].astype(str)
y = df['spam'].replace({0:"Not Spam",1:"Spam"}).astype("object")

# 划分训练集与测试集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# 词袋模型 word of bag处理文本数据，获得稀疏矩阵表示
count_vector = CountVectorizer(stop_words='english')

train_data = count_vector.fit_transform(x_train)

test_data = count_vector.transform(x_test)


# 朴素贝叶斯算法
start_time = time.time()
naive_bayes = MultinomialNB()
naive_bayes.fit(train_data, y_train)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
predictions_nb = naive_bayes.predict(test_data)
end_time = time.time()
print(f"朴素贝叶斯算法耗时: {end_time - start_time}")
print('朴素贝叶斯算法精确度:', format(accuracy_score(y_test, predictions_nb)))

# 支持向量机模型对数据进行拟合
start_time = time.time()
svm_clf = svm.SVC(gamma='scale')
svm_clf.fit(train_data, y_train)
predictions_svm = svm_clf.predict(test_data)
end_time = time.time()
print(f"支持向量机耗时: {end_time - start_time}")
print('支持向量机精确度:', format(accuracy_score(y_test, predictions_svm)))

# ID3算法
start_time = time.time()
decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_data, y_train)
predictions_dt = decision_tree.predict(test_data)
end_time = time.time()
print(f"ID3算法耗时: {end_time - start_time}")
print('ID3算法精确度', format(accuracy_score(y_test, predictions_dt)))





