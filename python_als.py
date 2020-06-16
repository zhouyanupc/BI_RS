from ALS import *

print("使用ALS算法")
model = ALS()
# 数据加载
X = load_movie_ratings('./ratings_small.csv')
# 运行max_iter次
model.fit(X, k=3, max_iter=2)
"""
X = np.array([[1,1,1], [1,2,1], [2,1,1], [2,3,1], [3,2,1], [3,3,1], [4,1,1], [4,2,1],
              [5,4,1], [5,5,1], [6,4,1], [6,6,1], [7,5,1], [7,6,1], [8,4,1], [8,5,1], [9,4,1], [9,5,1],
              [10,7,1], [10,8,1], [11,8,1], [11,9,1], [12,7,1], [12,9,1]])
# 运行max_iter次
model.fit(X, k=3, max_iter=20)
"""

print("对用户进行推荐")
user_ids = range(1, 13)
# 对用户列表user_ids，进行Top-n推荐
predictions = model.predict(user_ids, n_items=2)
print(predictions)
for user_id, prediction in zip(user_ids, predictions):
    _prediction = [format_prediction(item_id, score) for item_id, score in prediction]
    print("User id:%d recommedation: %s" % (user_id, _prediction))