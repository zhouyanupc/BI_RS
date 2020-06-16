# 数据清洗 和 预测
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score
import numpy as np

# 数据加载
train_data = pd.read_csv('titanic/train.csv')
test_data = pd.read_csv('titanic/test.csv')
# 数据探索
# 查看train_data信息
pd.set_option('display.max_columns', None)
print('查看数据信息:列名/非空个数/类型/等')
print(train_data.info())
print('*'*50)
print('查看数据摘要')
print(train_data.describe())
print('*'*50)
print('查看离散数据分布')
print(train_data.describe(include=['O']))
print('*'*50)
print('查看head')
print(train_data.head())
print('*'*50)
print('查看tail')
print(train_data.tail())

# 使用平均年龄填充Age缺失值
train_data['Age'].fillna(train_data['Age'].mean(),inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)

# 使用票价均值填充Fare缺失值
train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)

# 使用众数来填充Embark缺失值
print(train_data['Embarked'].value_counts())
train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S', inplace=True)

# 特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]
print('特征值')
print(train_features)

dvec = DictVectorizer(sparse=False)
train_features = dvec.fit_transform(train_features.to_dict(orient='record'))
print(dvec.feature_names_)

# ID3决策树
clf = DecisionTreeClassifier(criterion='entropy')
# 决策树训练
clf.fit(train_features, train_labels)

test_features = dvec.transform(test_features.to_dict(orient='record'))
# 决策树预测
pred_labels = clf.predict(test_features)
# 决策树准确率
acc_decision_tree = round(clf.score(train_features, train_labels), 6)
print(u'决策树准确率为 %.4lf' % acc_decision_tree)

# 使用K折交叉验证
print(u'cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(clf, train_features, train_labels, cv=10)))
