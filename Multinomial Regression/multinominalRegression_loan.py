# -*- coding: utf-8 -*-
"""
You work for a consumer finance company which specializes in lending 
loans to urban customers. When the company receives a loan application,
 the company has to make a decision for loan approval based on the 
 applicant’s profile. Two types of risks are associated with the bank’s decision: 
• If the applicant is likely to repay the loan, 
then not approving the loan results in a loss of business to the company 
• If the applicant is not likely to repay the loan, 
i.e. he/she is likely to default, then approving 
the loan may lead to a financial loss for the company 

The data given below contains the information about past loan applicants and whether they ‘defaulted’4 or not. The aim is to identify patterns which indicate if a person is likely to default, which may be used for taking actions such as denying the loan, reducing the amount of loan, lending (to risky applicants) at a higher interest rate, etc. 

In this case study, you will use EDA to understand how consumer attributes and loan attributes influence the tendency of default. 

When a person applies for a loan, there are two types of decisions that could be taken by the company: 

1. Loan accepted: If the company approves the loan, there are 3 possible scenarios described below: 
o Fully paid: Applicant has fully paid the loan (the principal and the interest rate) 
o Current: Applicant is in the process of paying the instalments, i.e. the tenure of the loan is not yet completed. These candidates are not labelled as 'defaulted'. 
o Charged-off: Applicant has not paid the instalments in due time for a long period of time, i.e. he/she has defaulted on the loan  
2. Loan rejected: The company had rejected the loan (because the candidate does not meet their requirements etc.). Since the loan was rejected, there is no transactional history of those applicants with the company and so this data is not available with the company (and thus in this dataset)

This company is the largest online loan marketplace, facilitating personal loans, business loans, and financing of medical procedures. Borrowers can easily access lower interest rate loans through an online interface.  
 Like most other lending companies, lending loans to ‘risky’ applicants is the largest source of financial loss (called credit loss). The credit loss is the amount of money lost by the lender when the borrower refuses to pay or runs away with the money owed. In other words, borrowers who default cause the largest amount of loss to the lenders. In this case, the customers labelled as 'charged-off' are the 'defaulters'.  
 If one is able to identify these risky loan applicants, then such loans can be reduced thereby cutting down the amount of credit loss. 
In other words, the company wants to understand the driving factors (or driver variables) behind loan default, i.e. the variables which are strong indicators of default.  The company can utilize this knowledge for its portfolio and risk assessment.  

Perform Multinomial regression on the dataset in which loan_status is the output (Y) variable and it has three levels in it. 


1.	Business Problem
1.1.	What is the business objective?
        Basic understanding of risk analytics in banking and financial services and understand how data is used to minimise the risk of losing money while lending to customers.
        Development of multinomial model to suggest to sanction loan or not
1.2.	Are there any constraints?
        Analysis has to be done on huge parameters,extracting desired parameters is real challenge
@author: Radhakrishna Naik
"""
#Data set after feature engineering
#loan_status                     int32
#id                              int64
#member_id                       int64
#loan_amnt                       int64
#funded_amnt_inv               float64
#term                            int32
#int_rate                        int32
#installment                   float64
#grade                           int32
#sub_grade                       int32
#emp_title                     float64
#emp_length                      int32
#home_ownership                  int32
#annual_inc                    float64
#verification_status             int32
#issue_d                         int32
#pymnt_plan                      int32
#purpose                         int32
#dti                           float64
#initial_list_status             int32
#collections_12_mths_ex_med    float64
#policy_code                     int64
#acc_now_delinq                  int64
#chargeoff_within_12_mths      float64
#delinq_amnt                     int64
#pub_rec_bankruptcies          float64
#tax_liens                     float64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
loan=pd.read_csv("c:/360DG/Datasets/loan.csv")
loan.dtypes
loan.head(10)
loan.columns
#Some of the important columns in the dataset are loan_amount, term, interest rate, grade, sub grade, annual income, purpose of the loan etc.
#The target variable, which we want to compare across the independent variables, is loan status.
loan.isnull().sum()
#There are several columns having missing values at higher level
# removing the columns having more than 90% missing values
missing_columns = loan.columns[100*(loan.isnull().sum()/len(loan.index)) > 90]
print(missing_columns)
loan = loan.drop(missing_columns, axis=1)
print(loan.shape)
# summarise number of missing values again
100*(loan.isnull().sum()/len(loan.index))
#There are desc and mths_since_last_delinq columns having missing values 32.58
#64.66 % let us drop these two columns
# dropping the two columns
loan = loan.drop(['desc', 'mths_since_last_delinq'], axis=1)
print(loan.shape)
# summarise number of missing values again
100*(loan.isnull().sum()/len(loan.index))
#Now let us impute these missing values
#emp_title,emp_length,title,revol_util
loan.dtypes
#grade,sub_grade,emp_title,emp_length,home_ownership,verification_status
#issue_d,loan_status,pymnt_plan,url,purpose,title,zip_code,addr_state,earliest_cr_line
#revol_util,initial_list_status,last_pymnt_d,last_credit_pull_d,application_type are of object type
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
loan.grade=labelencoder.fit_transform(loan.grade)
loan.sub_grade=labelencoder.fit_transform(loan.sub_grade)
loan.emp_title=labelencoder.fit_transform(loan.emp_title)
loan.emp_length=labelencoder.fit_transform(loan.emp_length)
loan.home_ownership=labelencoder.fit_transform(loan.home_ownership)
loan.verification_status=labelencoder.fit_transform(loan.verification_status)
loan.issue_d=labelencoder.fit_transform(loan.issue_d)
loan.loan_status=labelencoder.fit_transform(loan.loan_status)
loan.pymnt_plan=labelencoder.fit_transform(loan.pymnt_plan)
loan.url=labelencoder.fit_transform(loan.url)
loan.purpose=labelencoder.fit_transform(loan.purpose)
loan.title=labelencoder.fit_transform(loan.title)
loan.zip_code=labelencoder.fit_transform(loan.zip_code)
loan.addr_state=labelencoder.fit_transform(loan.addr_state)
loan.earliest_cr_line=labelencoder.fit_transform(loan.earliest_cr_line)
loan.revol_util=labelencoder.fit_transform(loan.revol_util)
loan.initial_list_status=labelencoder.fit_transform(loan.initial_list_status)
loan.last_pymnt_d=labelencoder.fit_transform(loan.last_pymnt_d)
loan.last_credit_pull_d=labelencoder.fit_transform(loan.last_credit_pull_d)

loan.term=labelencoder.fit_transform(loan.term)
loan.int_rate=labelencoder.fit_transform(loan.int_rate)
loan.application_type=labelencoder.fit_transform(loan.application_type)
loan.dtypes

##############3
# summarise number of missing values again
100*(loan.isnull().sum()/len(loan.index))
#
import numpy as np
from sklearn.impute import SimpleImputer
mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
loan['emp_title']=pd.DataFrame(mean_imputer.fit_transform(loan[["emp_title"]]))
loan['collections_12_mths_ex_med']=pd.DataFrame(mean_imputer.fit_transform(loan[["collections_12_mths_ex_med"]]))
loan['chargeoff_within_12_mths']=pd.DataFrame(mean_imputer.fit_transform(loan[["chargeoff_within_12_mths"]]))
loan['pub_rec_bankruptcies']=pd.DataFrame(mean_imputer.fit_transform(loan[["pub_rec_bankruptcies"]]))
loan['tax_liens']=pd.DataFrame(mean_imputer.fit_transform(loan[["tax_liens"]]))
loan.isnull().sum()
#################################################
#In the given data set there are Customer behaviour variables 
#(those which are generated after the loan is approved such as delinquent 2 years, revolving balance, next payment date etc.).
#They are not really required for the saction of the loan
#we need variables like 1. those which are related to the applicant 
#(demographic variables such as age, occupation, employment details etc.),
# 2. loan characteristics (amount of loan, interest rate, purpose of 
#loan etc.)
behaviour_var =  [
  "delinq_2yrs",
  "earliest_cr_line",
  "inq_last_6mths",
  "open_acc",
  "pub_rec",
  "revol_bal",
  "revol_util",
  "total_acc",
  "out_prncp",
  "out_prncp_inv",
  "total_pymnt",
  "total_pymnt_inv",
  "total_rec_prncp",
  "total_rec_int",
  "total_rec_late_fee",
  "recoveries",
  "collection_recovery_fee",
  "last_pymnt_d",
  "last_pymnt_amnt",
  "last_credit_pull_d",
  "application_type"]
behaviour_var
# let's now remove the behaviour variables from analysis
loan = loan.drop(behaviour_var, axis=1)
loan.dtypes
#Typically, variables such as acc_now_delinquent, chargeoff within 12 months etc. (which are related to the applicant's past loans) are available from the credit bureau.
# also, we will not be able to use the variables zip code, address, state etc.
# the variable 'title' is derived from the variable 'purpose'
# thus let get rid of all these variables as well
loan = loan.drop(['title', 'url', 'zip_code', 'addr_state'], axis=1)
loan.dtypes
#our target variable is loan_status,let us shift to 0 th position
loan=loan.iloc[:,[16,0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26,27]]
#There are several columns having different scale,let us go for normalization
#loan.loc[loan.loan_status>1,"loan_status"]=1
loan['loan_status'].value_counts()
#Fully Paid     32950
#Charged Off     5627
#Current         1140
#fully paid comprises most of the loans. The ones marked 'current' are neither fully paid not defaulted, so let's get rid of the current loans. Also, let's tag the other two values as 0 or 1.
#let us drop current account because they are of no use
# filtering only fully paid or charged-off
loan = loan[loan['loan_status'] !=1]

loan.loc[loan.loan_status==2,"loan_status"]=1

# summarising the values
loan['loan_status'].value_counts()
#1    32950
#0     5626
########################################################
##########EDA

#First, let's look at the overall default rate.
#############################################
# default rate
default=5626/(32950+5626)
default
#The overall default rate is about 14%.
#Let's first visualise the average default rates across categorical variables.
# plotting default rates across grade of the loan
sns.barplot(x='grade', y='loan_status', data=loan)
plt.show()
#Grade 0 loans are highest default rate and gradually decreases to 6
#grade of loan goes from 0 to 6, the default rate increases. This is expected because the grade is decided by Lending Club based on the riskiness of the loan.
# lets define a function to plot loan_status across categorical variables
def plot_cat(cat_var):
    sns.barplot(x=cat_var, y='loan_status', data=loan)
    plt.show()
# term: 60 months loans default more than 36 months loans
plot_cat('term')
#60 months loan is greater than 30 months
# sub-grade: as expected - A1 is better than A2 better than A3 and so on 
plt.figure(figsize=(16, 6))
plot_cat('sub_grade')
round(np.mean(loan['loan_status']),1)
# home ownership: not a great discriminator
plot_cat('home_ownership')
#0-rent,1=own,2=Mortgage,3=other
# verification_status: surprisingly, verified loans default more than not verifiedb
plot_cat('verification_status')
#0=verfied,1=source verified and 2=not verified
# purpose: small business loans defualt the most, then renewable energy and education
plt.figure(figsize=(16, 6))
plot_cat('purpose')
#6=small business loan which is higher
# loan amount: the median loan amount is around 10,000
sns.distplot(loan['loan_amnt'])
plt.show()
#median of loan amount is 10000
# comparing default rates across rates of interest
# high interest rates default more, as expected
loan.int_rate.describe()
#average interest rate is 179.17
#min is 0
#max is 370
# annual income and default rate
# purpose: small business loans defualt the most, then renewable energy and education
plt.figure(figsize=(16, 6))
plot_cat('purpose')
#We have now compared the default rates across various variables, and some of the important predictors are purpose of the loan, interest rate, annual income, grade etc.
#In the credit industry, one of the most important factors affecting default is the purpose of the loan - home loans perform differently than credit cards, credit cards are very different from debt condolidation loans etc.
# variation of default rate across annual_inc
loan.groupby('annual_inc').loan_status.mean().sort_values(ascending=False)
########################################

tc = loan.corr()
tc
import matplotlib.pyplot as plt
import seaborn as sns
fig,ax= plt.subplots()
fig.set_size_inches(200,10)
sns.heatmap(tc, annot=True, cmap='YlGnBu')
#loan_status is highly correlated with interest rate
#let us split the data
train,test=train_test_split(loan,test_size=0.2)
#multinomial option is only supported the 'lbfgs','newton rapson -cg'solver
model=LogisticRegression(multi_class="multinomial",solver="newton-cg").fit(train.iloc[:,1:],train.iloc[:,0])
test_pred=model.predict(test.iloc[:,1:])
####
accuracy_score(test.iloc[:,0],test_pred)
#Train predict
train_pred=model.predict(train.iloc[:,1:])
accuracy_score(train.iloc[:,0],train_pred)
#6.	Write about the benefits/impact of the solution - in what way does the business (client) benefit from the solution provided?
#Analysing various factors on loan default,one can easily decide ,wheather to sanction loan or not.
loan.dtypes
