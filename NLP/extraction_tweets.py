# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 17:22:53 2022

@author: Dell
"""


import pandas as pd
import tweepy
from tweepy import OAuthHandler
consumer_key=" 6kJetKnJXfuSzsalnEhSLoFCN "
consumer_secret=" Y3dZyl1X6xMiLiR3HXiblS2ap033xzlcOvPvMENQbBck9Kanlik"
access_token="  68933593-syZvk4hnHkonf76ryk8SaoZgXcddpEko6oaol"
access_token_secret=" NBQv9dRZAqHvlyNpOH2s909INJFQMmS6j8VRQJC3D1UOJ"
auth=tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)  
api=tweepy.API(auth)
keyword='Ukrain'
tweets_keyword=api.search_tweets(keyword,count=100)
for item in tweets_keyword:
    print(item)
tweets_for_csv=[tweet.full_text for tweet in tweets_keword]
################################################################
tweets_user=api.user_timeline(screen_name="ShashiTharoor",count=200)

# read the contents
for item in tweets_user:
    print(item)
tweets_for_csv1=[tweet.full_text for tweet in tweets_keword]   
    
