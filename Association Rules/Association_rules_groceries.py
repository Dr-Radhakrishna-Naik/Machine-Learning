# -*- coding: utf-8 -*-
"""
The Departmental Store, has gathered the data of the products it sells on a Daily basis.
Using Association Rules concepts, provide the insights on the rules and the plots.


Created on Fri Apr  8 14:22:13 2022

@author: Radhakrishna Naik
"""

from mlxtend.frequent_patterns import apriori,association_rules
#Here we are going to use transactional data wherein size of each row is not consistent
#We can not use pandas to load this unstructured data
#here fuction called open() is used
#create an empty list
groceries=[]
with open("c:/360DG/Assignments/Association Rules/groceries.csv") as f:transactions_retail1=f.read()
#splitting the data into seperate transactions using seperator,it is comma seperated contents
#we can use new line character "\n"
groceries=transactions_retail1.split("\n")
#Earlier retail datastructure was in string format,now it will change to list of
#557042 ,each item is comma seperated
# our main aim is to calculate #A ,#C,we will have to seperate out each item from each transaction
groceries_list=[]
for i in groceries:
    groceries_list.append(i.split(","))
# split fuction will seperate each item from each list,wherever it will find comma ,it will split the item
# in order to generate association rules ,you can directly use retail_list
# Now let us seperate out each item from the retail list
all_groceries_list=[ i for item in groceries_list for i in item]    
# You will get all the items occured in all transactions
#We will get 3348059 items in various transactions

#Now let us count the frequency of each item
#we will import collections package which has Counter function which will count the items
from collections import Counter
item_frequencies=Counter(all_groceries_list)
#item_frequencies is basically dictionary having x[0] as key and x[1]=values
# we want to access values and sort based on the count that occured in it. 
# it will show the count of each item purchased in eavery transactin
#Now let us sort these frquencies in ascending order
item_frequencies=sorted(item_frequencies.items(),key=lambda x:x[1])
item_frequencies
#when we execute this ,item frquencies will be in sorted form ,in the form of tuple
# item name with count
#Let us seperate out items and their count
items=list(reversed([i[0] for i in item_frequencies])) 
#This is list comprehension for each item in item frequencies access the key,i.e item
#here you will get items list
frequencies=list(reversed([i[1] for i in item_frequencies]))
#here you will get count of purchase of each item

import pandas as pd
#Now let us try to establish association rule mining
#we have groceries list in the list format,we need to convert it in dataframe format
groceries_series=pd.DataFrame(pd.Series(groceries_list))
# Now we will get dataframe of size 9836X1 size,coulmn comprises of multiple items
#we had extra row created,check the groceries_series ,last row is empty, let us first delete it
groceries_series=groceries_series.iloc[:9835,:]
#we have taken rows from 0 to 9834 and columns 0 to all
#groceries series has column having name 0,let us rename as transactions
groceries_series.columns=["Transactions"]
#Now we will have to apply 1-hot encoding,before that in one column there are various items seperated by ',
#','let us seperate it with'*'
x=groceries_series['Transactions'].str.join(sep='*')
#check the x in variable explorer which has * seprator rather the ','
x=x.str.get_dummies(sep='*')
#you will get one hot encoded dataframe of size 9835X169
#This is our input data to apply to apriori algorithm,it will generate !169 rules,min support value
#is 0.0075(it must be between 0 to1),you can give any number but must be between 0 and 1
frequent_itemsets=apriori(x,min_support=0.0040,max_len=4,use_colnames=True)
#you will get support values for 1,2,3 and 4 max items
#let us sort these support values
frequent_itemsets.sort_values('support',ascending=False,inplace=True)
#Support values will be sorted in descending order
#Even EDA was also have the same trend,in EDA there was count and here it is support value
#we will generate association rules,This association rule will calculate all the matrix
#of each and every combination
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=1) 
#This generate association rules of size 4472X5 columns comprizes of antescends,consequences
rules=rules[["antecedents","consequents","support","confidence","lift"]]
rules.sort_values('lift',ascending=False).head(10)
rules1=rules[rules['support']<0.04]
rules1
#The groceries which are sold 3 % minimum in single or in combination but with confidence=0.7,0.8,0.9 and 1
rules2=rules1[rules1['confidence']>0.7]
rules2
#1 if customer is buying root vegetables, yogurt then there is 70 % chances it will buy Whole milk
#
#2 if cutomer is buying butter, curd then there are 70 % chances that cutomer will buy whole milk
#if cutomer is buying domastic eggs, curd then there are 70 % chances that cutomer will buy whole milk

# so on and so fourth

