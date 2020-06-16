from pyspark.ml.recommendation import ALS
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
import pandas as pd
sc = SparkContext()
sql_sc = SQLContext(sc)

pd_df_ratings = pd.read_csv('ratings.csv')
pyspark_df_ratings = sql_sc.createDataFrame(pd_df_ratings)
pyspark_df_ratings = pyspark_df_ratings.drop('Timestamp')

# 创建ALS模型
als = ALS(rank=3, maxIter=10, regParam=0.1, userCol='userId',itemCol='movieId')
model = als.fit(pyspark_df_ratings)
# 对userId = 10 的用户进行Top-N 推荐
recommendations = model.recommendForAllUsers(5)
print(recommendations.where(recommendations.userId == 10).collect())