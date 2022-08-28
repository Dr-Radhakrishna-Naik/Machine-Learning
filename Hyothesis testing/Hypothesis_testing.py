# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 16:15:52 2022

@author: Dell
"""
import pandas as pd
import numpy as np
import scipy
from scipy import stats

import statsmodels.stats.descriptivestats as sd
from statsmodels.stats import Weightstats as stests
# 1 sample sign test
#Whenever there is single sample and data is not normal
marks=pd.read_csv("C:/360DG/Datasets/Signtest.csv")
#Normal QQ plot
import pylab
stats.probplot(marks.Scores,dist='norm',plot=pylab)
#Data is not normal
stats.shapiro(marks.Scores)
#p_value is 0.024<0.005,p is low null go
#H0-data is normal
#H1-data is not normal
#H0 is not valid ,i.e data is not normal
#Let us check the distribution of the data
marks.Scores.describe()
#1 sample sign test
sd.sign_test(marks.Scores,mu0=marks.Scores.mean())
#p_value is 0.82>0.05 so p is high null fly
#H0 =scores are either equal or less than 80
#H1=scores are not equal n greater than 80
##################################################
#1 sample z test
fabric=pd.read_csv("C:/360DG/Datasets/Fabric_data.csv")
#check for normality test
#H0=Data is normal
#H1=Data is not normal
print(stats.shapiro(fabric))
#p_value is 0.14>0.05 hance p is high null fly
#data is normal
np.mean(fabric)
#155.064
#H0=fabric length is less than 150

#H1=Fabric length is more than 150
ztest,pval=stets.ztest(fabric,x2=none,value=155.064)
#p_value is 7.15x10^-6<0.05 p is low n null go
#########################################
#Mann Whitsney test
#We have to aidentify the impact of additive in fuel,does it 
#improve the performance or not
#Whenever there are two samples and one of them is not normal then
#Man_Whitsney test is applied
fuel=pd.read_csv("c:/360DG/Datasets/mann_whitney_additive.csv")
fuel.columns="Without_additive","With_additive"
#Normality test
#H0= data is normal
#H1=data is not normal
print(stats.shapiro(fuel.Without_additive))
# pvalue=0.501 > 0.05 hanece data is normal
print(stats.shapiro(fuel.With_additive))
#pvalue=0.04 is < than 0.05 hence data is not normal
#Let us apply Mann_Whitney test
#H0= fuel additive does not impact the performance
#H1=fuel additive does impact on the performance
scipy.stats.mannwhitneyu(fuel.Without_additive,fuel.With_additive)
#pvalue=0.44573,p is high null fly
#H0= fuel additive does not impact the performance
######################################################
#paired T-test
#Whever there are two samples and both are normally distributed
#Their external conditions are same then paired_T_test is applied
sup=pd.read_csv("C:/360DG/Datasets/paired2.csv")
#Here we have to check which supplier has less transaction time
#i.e.time taken to complete back office 
#normality test
#H0=data is normal
#H1=data is not normal
stats.shapiro(sup.SupplierA)
#pvalue=0.8961 >0.05 hence H0 is true
stats.shapiro(sup.SupplierB)
#pvalue=0.8961 >0.05 hence h0 is true
#Let us apply paired T-test,assuming external conditions are same
#H0=There is no significant difference between A and B
#H1=There is significant difference between A and B
ttest,pval=stats.ttest_rel(sup['SupplierA'],sup['SupplierB'])
pval
#0<0.05 H1 is true,that there is significant difference between A and B
##################################################
#when there are two samples,they are normal but their external conditions are not
#same then 2_Sample_T_test is used
#Here there are two offers are their for credit cards,we have to find
#whether both offers are equal or different
offers=pd.read_excel("c:/360DG/Datasets/promotion.xlsx")
offers.columns="InterestRateWaiver","StandardPromotion"
#Let us check the normality
#H0=data is normal
#H1=Data is not normal
stats.shapiro(offers.InterestRateWaiver)
#pvalue=0.2245>0.05 p high null fly
stats.shapiro(offers.StandardPromotion)
#pvalue=0.1915 > 0.05 p high ,data is normal
#Variance test
#H0=Equal variance
#H1=not equal variance
scipy.stats.levene(offers.InterestRateWaiver,offers.StandardPromotion)
#pvalue=0.287 >0.05 both have equal variance
##let us apply two sample T test
#H0= both offers are equal
#H1=both offers are different
scipy.stats.ttest_ind(offers.InterestRateWaiver,offers.StandardPromotion)
#pvalue=0.0242 <0.05 H1 is true
############################################
#if there are more than two samples then you have to check the normality test
#if either of the sample is not normal then go for Moods Median Test
import pandas as pd
animals=pd.read_csv("C:/360DG/Datasets/animals.csv")
#Check for normality test
#H0=data is normal
#H1=Data is not normal
stats.shapiro(animals.Pooh)
#pvalue=0.0122<0.05 H1 is true
stats.shapiro(animals.Piglet)
#pvalue=0.044 <0.05 H1 is true
stats.shapiro(animals.Tigger)
#pvalue=0.02194<0.05 H1 is true
#Now let us apply moods median Test
#H0=all the animals are equal in number
#H1=not equal
from scipy.stats import median_test
stats,pval,med,tb1=median_test(animals.Pooh,animals.Piglet,animals.Tigger)
pval
#0.18637 >0.05 p high null fly
##########################################################
#When there are more than 2 samples and all the samples are normally distributed
#having equal variance then one way ANOVA test is used
contract=pd.read_excel("c:/360DG/Datasets/ContractRenewal_Data(unstacked).xlsx")
contract.columns="Supp_A","Supp_B","Supp_C"
#Check for normality test
#H0=data is normal
#H1=Data is not normal
stats.shapiro(contract.Supp_A)
#pvalue=0.8961>0.05 ,data is normal
stats.shapiro(contract.Supp_B)
#pvalue=0.6483 >0.05 data is normal
stats.shapiro(contract.Supp_C)
#pvalue=0.57190 > 0.05 data is normal
# variance test
#H0=data is equally variant
#H1=Data is not equally varient
scipy.stats.levene(contract.Supp_A,contract.Supp_B,contract.Supp_C)
#pvalue=0.7775 >0.05 data is equally varient
#One way Anova test
#H0=all three suppliers are having equal transaction time
#H1=all three suppliers are not  having equal transaction time
F,pval=stats.f_oneway(contract.Supp_A,contract.Supp_B,contract.Supp_C)
pval
#0.103732>0.05 hence H0 is true
################################################
##################################################
#Two proportions test
#soft drink company want to launch sales promotion,want to know
#Either elders consume more than childrens
import numpy as np
soft_drink=pd.read_excel("c:/360DG/Datasets/JohnyTalkers.xlsx")
from statsmodels.stats.proportion import proportions_ztest
tab1=soft_drink.Person.value_counts()
tab1
tab2=soft_drink.Drinks.value_counts()
tab2
pd.crosstab(soft_drink.Person,soft_drink.Drinks)
#How many users purchased
#We have total 1220 out of which 210 are purchased and 1010 not purchased
#out of 210 58 adults purchased and 152 by children

#0.1037>0.05
count = np.array([58, 152])
#we have 1220 persons out of which 480 are adults and 740 are childrens
nobs = np.array([480, 740])


stats, pval = proportions_ztest(count, nobs, alternative = 'two-sided') 
print(pval) # Pvalue 0.000

stats, pval = proportions_ztest(count, nobs, alternative = 'larger')
print(pval)  # Pvalue 0.999  
##########################################################
#Chi square test
Bahaman = pd.read_excel("C:/360DG/Datasets/Bahaman.xlsx")
Bahaman

count = pd.crosstab(Bahaman["Defective"], Bahaman["Country"])
count
Chisquares_results = scipy.stats.chi2_contingency(count)

Chi_square = [['Test Statistic', 'p-value'], [Chisquares_results[0], Chisquares_results[1]]]
Chi_square
#Proportions of deffective in each country equal
#proportions of deffective in each county is not equal.
