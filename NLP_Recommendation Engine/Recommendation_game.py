# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 15:23:38 2022

@author: Dell
"""
import pandas as pd
game=pd.read_csv("c:/360DG/Datasets/game.csv",encoding='utf8')
game.shape
#you will get 12294X7 matrix
game.columns
game.userId
game.dtypes
game['userId'].isnull().sum()
game['game'].isnull().sum()
game.describe()
 voters=5000
 ave_voters=3432.28
 ave_vote=3.59
 mini_vote=0.5
#minimum of rating=0.50
# filter the game into another dataframe applying filter condition of rating
game1=game.copy().loc[game['rating']>0.5]
def weighted_rating(voters,ave_voters,mini_vote,ave_vote):
    d1=(voters/(voters+mini_vote))*ave_vote
    d2=(mini_vote/(voters+mini_vote))*ave_voters
    wt_rate=d1+d2
    return wt_rate
wt_rate=weighted_rating(voters,ave_voters,ave_vote,mini_vote)
game1['score']=game1.apply(wt_rate,axis=1)
game1=game1.sort_values('rating',ascending=False)

game1.head(10)
