#    author：酱油
#    time：2020-06-01
#

import pandas as pd
import numpy as np


def weight_rating(x,m=m,C=C):
   """计算IMDB"""
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m)*R)+(m/(v+m)*C)

#加载数据
df_credits = pd.read_csv('./TMDB_Movie/tmdb_5000_credits.csv')
df_movies = pd.read_csv('./TMDB_Movie/tmdb_5000_movies.csv')

#合并数据
df_credits.columns=['id','tittle','cast','crew']
df_data = df_movies.merge(df_credits,on='id')

#首先我们需要如下准备：
#我们需要一个指标来给电影评分
#计算每部电影的分数
#按照得分排序并向用户推荐得分最高的电影
#使用IMDB计算电影评分(wr),
#𝑊𝑅=(𝑣/(𝑣+𝑚)𝑅)+(𝑚/(𝑣+𝑚)𝐶)
#v 电影的评分人数;
#m 系统要求最低的电影的评分人数;
#R 电影平均评分
#C 所有电影的平均评分

C = df_data['vote_average'].mean()
# m 指的是大于m个评论人的电影
m = df_data['vote_count'].quantile(0.9)

q_movies = df_data.copy().loc[df_data['vote_count']>=m]
q_movies['score']=q_movies.apply(weight_rating,axis=1)

# 获取score高的电影
q_movies=q_movies.sort_values('score',ascending=False)
q_movies = q_movies[['title', 'vote_count', 'vote_average', 'score']]
#取top10
q_movies[:10]




