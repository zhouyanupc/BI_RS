from surprise import Dataset
from surprise import Reader
from surprise import BaselineOnly, KNNBasic, NormalPredictor
from surprise import accuracy
from surprise.model_selection import KFold

# 数据读取
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file('ratings.csv', reader=reader)
train_set = data.build_full_trainset()

# ALS优化
bsl_options = {'method': 'als', 'n_epochs': 5, 'reg_u': 12, 'reg_i': 5}
# SGD优化
# bsl_options = {'method': 'sgd', 'n_epochs': 5}
algo = BaselineOnly(bsl_options=bsl_options)
# algo = KNNBasic(bsl_options=bsl_options)
# algo = NormalPredictor()

# 使用K折交叉验证
kf = KFold(n_splits=3)
for train_set, test_set in kf.split(data):
    # 训练并预测
    algo.fit(train_set)
    predictions = algo.test(test_set)
    # 计算RMSE
    accuracy.rmse(predictions, verbose=True)

# 输出uid对iid的预测结果
pred = algo.predict(uid=str(196), iid=str(302), r_ui=4, verbose=True) # r_ui = 4.00   est = 4.06