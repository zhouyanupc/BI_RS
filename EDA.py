import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from matplotlib.font_manager import FontProperties

# 散点图
# N = 500
# x = np.random.randn(N)
# y = np.random.randn(N)
# # 用Matplotlib画散点图
# plt.scatter(x,y,marker='x')
# plt.show()
# # 用Seaborn画散点图
# df = pd.DataFrame({'x':x, 'y':y,})
# sns.jointplot(x='x', y='y', data=df, kind='scatter')
# plt.show()

# # 折线图
# x = [1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910]
# y = [265, 323, 136, 220, 305, 350, 419, 450, 560, 720, 830]
# # 用Matplotlib画散点图
# plt.plot(x,y)
# plt.show()
# # 用Seaborn画散点图
# df = pd.DataFrame({'x':x, 'y':y})
# sns.lineplot(x='x', y='y', data=df)
# plt.show()

# # 条形图
# x = ['c1', 'c2', 'c3', 'c4']
# y = [15, 18, 5, 26]
# # 用Matplotlib画条形图
# plt.bar(x,y)
# plt.show()
# # 用Seaborn画条形图
# sns.barplot(x,y)
# plt.show()

# # 箱线图
#
# data = np.random.normal(size=(20,4))
# labels = ['A','B','C','D']
# print(data)
# # 用Matplotlib画箱线图
# plt.boxplot(data, labels=labels)
# plt.show()
# # 用Seaborn画箱线图
# df = pd.DataFrame(data,columns=labels)
# sns.boxplot(data=df)
# plt.show()

# # 画饼图 只有matplotlib有饼图
# nums = [25, 33, 37]
# labels = ['ADC', 'APC', 'TK']
# plt.pie(x=nums, labels=labels)
# plt.show()
# # 另一种方式
# data = {'ADC':25, 'APC':33, 'TK':37}
# data = pd.Series(data)
# data.plot(kind="pie", labels=labels)
# plt.show()

# 热力图
np.random.seed(30)
data = np.random.rand(20,20)
print(data)
heat_map = sns.heatmap(data)
plt.show()