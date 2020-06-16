from surprise import Reader, Dataset
from surprise import SlopeOne

# 数据读取
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file('ratings.csv',reader=reader)
train_set = data.build_full_trainset()

# 使用slopeone算法
algo = SlopeOne()
algo.fit(train_set)
# 对指定用户和商品进行评分预测
uid = str(196)
iid = str(302)
pred = algo.predict(uid, iid, r_ui=4, verbose=True) # r_ui = 4.00   est = 4.32