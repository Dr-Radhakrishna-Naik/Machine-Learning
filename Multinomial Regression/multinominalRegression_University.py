# -*- coding: utf-8 -*-
"""
Multinomial regression working on University data to classify whether a candidate would be opting or being selected for Academic, General or Vocational

Problem Statement:

A University would like to effectively classify their students based on the program they are enrolled in. Perform multinomial regression on the given dataset and provide insights (in the documentation).

a. prog: is a categorical variable indicating what type of program a student is in: “General” (1), “Academic” (2), or “Vocational” (3).

b. Ses: is a categorical variable indicating someone’s socioeconomic status: “Low” (1), “Middle” (2), and “High” (3).

c. read, write, math, and science are their scores on different tests.

d. honors: Whether they are an honor roll or not.
1.	Business Problem
1.1.	What is the business objective?
        Classifying which program a student is likely to select or opt for based on the various variables given.
1.2.	Are there any constraints?
        Analysis has to be done on given parameters,but there are several other parameters affecting on admission
@author: Radhakrishna Naik
"""
#Unnamed: 0     int64
#id             int64
#female        object
#ses           object
#schtyp        object
#prog          object
#read           int64
#write          int64
#math           int64
#science        int64
#honors        object
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
opt=pd.read_csv("c:/360DG/Datasets/mdata.csv")
opt.dtypes
#first two columns are not really useful hence deleted
opt=opt.drop(["Unnamed: 0","id"],axis=1)
#let us check,how many students are  of General , Academic  and Vocation are 50
opt.prog.value_counts()
#There are 105 Academic,50 are vocational and 45 are general
#Let us check how many are male and females
opt.female.value_counts()
#There are 109 females and 91 are male
#let us check Ses: is a categorical variable indicating someone’s socioeconomic status: “Low” (1), “Middle” (2), and “High” (3).
opt.ses.value_counts()
#middle    95
#high      58
#low       47
#prog: is a categorical variable indicating what type of program a student is in: “General” (1), “Academic” (2), or “Vocational” (3).
opt.prog.value_counts()
#academic    105
#vocation     50
#general      45
#Let us check how many have opted for honors
opt.honors.value_counts()
#not enrolled    147
#enrolled         53
opt.describe()
#average reading test score is 52.23 ,min=28 and max is 76
#average writing test score is 52.77 ,min=31 and mx=67
#average math test score is 52.64 ,min=33 and max is 75
#avaerage science score is 51.85 and min=26 and max is 74
#Let us check how many male and female qualified read,write,math and science test
sns.boxplot(x="female",y="read",data=opt)
#average of male qualifying read test is more than ave of female
sns.boxplot(x="female",y="write",data=opt)
#However average qualifying female  is more than male 
sns.boxplot(x="female",y="math",data=opt)
#average qualifying female  is more than male in math test
sns.boxplot(x="female",y="science",data=opt)
#but average qualifying female  is less than male in science test
#Now let us check the how many students of various background are qualified the test
sns.boxplot(x="ses",y="read",data=opt)
#average of higher category students qualifying read test is more than ave of low and middle
sns.boxplot(x="ses",y="write",data=opt)
#average of higher category students qualifying read test is more than ave of low and middle
sns.boxplot(x="ses",y="math",data=opt)
#average of higher category students qualifying read test is more than ave of low and middle
sns.boxplot(x="ses",y="science",data=opt)
#average of higher category students qualifying read test is more than ave of low and middle
opt.dtypes
#female,ses,schtyp,prog and honors is object type
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
opt.female=labelencoder.fit_transform(opt.female)
opt.ses=labelencoder.fit_transform(opt.ses)
opt.schtyp=labelencoder.fit_transform(opt.schtyp)
opt.prog=labelencoder.fit_transform(opt.prog)
opt.honors=labelencoder.fit_transform(opt.honors)
#There are several columns having different scale,let us go for normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
opt1=norm_func(opt.iloc[:,:])
opt1.isna().sum()

opt1.loc[opt1.prog<1,"prog"]=0
#Let us re-arrange the columns
opt1=opt1.iloc[:,[3,0,1,2,4,5,6,7,8]]
tc = opt1.corr()
tc
import matplotlib.pyplot as plt
import seaborn as sns
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(tc, annot=True, cmap='YlGnBu')
#ses,schtyp and honors are strongly correlated
#let us split the data
train,test=train_test_split(opt1,test_size=0.2)
#multinomial option is only supported the 'lbfgs','newton rapson -cg'solver
model=LogisticRegression(multi_class="multinomial",solver="newton-cg").fit(train.iloc[:,1:],train.iloc[:,0])
test_pred=model.predict(test.iloc[:,1:])
####
accuracy_score(test.iloc[:,0],test_pred)
#Train predict
train_pred=model.predict(train.iloc[:,1:])
accuracy_score(train.iloc[:,0],train_pred)
#6.	Write about the benefits/impact of the solution - in what way does the business (client) benefit from the solution provided?
#Classifying which program a student is likely to select or opt for based on the various variables given.
