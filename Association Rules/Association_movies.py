# -*- coding: utf-8 -*-
"""
Kitabi Duniya, a famous book store in India,
 which was established before Independence,
 the growth of the company was incremental year by year,
 but due to online selling of books and wide spread Internet access its annual growth started to collapse,
 seeing sharp downfalls, you as a Data Scientist help this heritage book store 
 gain its popularity back and increase footfall of customers and provide ways the business can improve exponentially, apply Association RuleAlgorithm, explain the rules, and visualize the graphs for clear understanding of solution.
Created on Fri Apr  8 14:22:13 2022
1.1.	What is the business objective?
The main objective is to create a association rules to recommend relevant books to seller based on support,confidence and lift.
In addition to the Association rules Model prediction, we also have taken into account the  recommendation for a sale increase to a book seller.

1.2.	Are there any constraints?
Understanding the metric for evaluation was a challenge as well.
Since the data consisted of binary data, EDA of  was a major challenge..
Lastly we can not find which books are sold most
@author: Radhakrishna Naik
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
#let us import the book data set
movies=pd.read_csv("c:/360DG/Datasets/my_movies.csv")

frequent_itemsets=apriori(movies,min_support=0.03,max_len=4,use_colnames=True)
#items or item set must have minimum support value 3 % sale i.e.0.03
#you will get support values for 1,2,3 and 4 max items
#let us sort these support values
frequent_itemsets.sort_values('support',ascending=False,inplace=True)
#Support values will be sorted in descending order
#If we will  frequent_itemsets,70 % sale is of Gladiator,60 % sale is of sixth sense
#60% sale is of Patriot
#20% % sale is of Green Mile,LOTR1&LOTR2
#The lowest selling movies are Green Mile,LOTR1&LOTR2
#we will generate association rules,This association rule will calculate all the matrix
#of each and every combination
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=1) 
#This generate association rules of size 1198X9 columns comprizes of antescends,consequences
# let us generate books having  mini.3 % sale
rules1=rules[rules['confidence']>0.6]
rules1
#The books which are sold 3 % minimum in single or in combination but with confidence=0.7,0.8,0.9 and 1

#The books with confidence 70% genereate 0 rules
rules2=rules1[rules1['confidence']==0.8]
rules2
#
#1 if a person  is watching Patriot  then there is 100 % chances that person will watch Gladiator Moviee

#2 if a person  is watching sixth sense then there are 100 % chances that he will watch Gladiator
# so on so forth

