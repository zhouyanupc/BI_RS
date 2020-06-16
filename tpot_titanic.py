import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
import numpy as np

# 数据清洗
train_data = pd.read_csv('./titanic/train.csv').drop(['Name','Ticket','Cabin'], axis=1)

# 数据探索
pd.set_option('display.max_columns', None) # 显示所有列

print(train_data.info())
print('*'*50)
print(train_data.describe())
print('*'*50)
print(train_data.head())
print('*'*50)

# Age和Embarked数据缺失
# 用均值补全Age缺失值
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
# 用众数补全Embarked缺失值
print(train_data['Embarked'].value_counts())
train_data['Embarked'].fillna('S', inplace=True)

# 生成训练数据集
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']

dict_vec = DictVectorizer(sparse=False) # 不产生稀疏矩阵
train_features = dict_vec.fit_transform(train_features.to_dict(orient='record'))
X_train, X_test, y_train, y_test = train_test_split(train_features.astype(np.float64),
                                                    train_labels.astype(np.float64), train_size=0.75, test_size=0.25)
# TPOT自动机器学习
tpot = TPOTClassifier(generations=10, population_size=40, verbosity=2)
tpot.fit(X_train,y_train)
print(tpot.score(X_test,y_test))
tpot.export('tpot_titanic_pipeline.py')
