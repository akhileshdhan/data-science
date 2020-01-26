# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 10:12:22 2019

@author: Akhilesh Dhancholia
"""
#Import libraries for data analysis and data handling
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

#import libraries for EDA and VIZ
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#%matplotlib inline

# Setting the directory
import os
os.chdir('F:\Akhilesh - BABI\Capstone')

#read excel file
mydata=pd.read_csv('Taiwan-Customer defaults.csv')
mydata_raw=mydata

#summary of the variables
mydata.shape 
mydata.describe()
mydata.head(1)

#checking for null values
mydata.isnull().sum()
plt.plot(mydata.isnull().sum())

#Categorical Varibles Descripbtion
mydata[['SEX','MARRIAGE','EDUCATION']].describe()

#Payment Delay Description
mydata[['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5']].describe()

#BILL AMT Describtion
mydata[['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT5']].describe()

#Previous amount paid Description
mydata[['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']].describe()


## Data Preprocessing
#regrouping variables -- Eductaion and Marriage
mydata.loc[mydata.EDUCATION>=4 , 'EDUCATION']=4
mydata.loc[mydata.MARRIAGE==0,'MARRIAGE']=3

#renaming the variable "default payment next month" to DEFAULT
mydata=mydata.rename(columns={'default payment next month':'DEFAULT'})
 
#renaming the PAY_0 columns to PAY_1 to match with BILL_AMT
mydata=mydata.rename(columns={'PAY_0':'PAY_1'})

#print column names
mydata.columns

## Making a Count Plot
plt.figure(dpi=200)
sns.countplot(x='DEFAULT',hue='DEFAULT',data=mydata,palette='Set2')

#Finding the class of columns
print(mydata.dtypes)

#Converting DEFAULT into factors

mydata.DEFAULT=pd.Categorical(mydata['DEFAULT'])
print(mydata.DEFAULT.dtype)

## Exploration Data Analysis (EDA)

#Univariate Analysis

#Grouping based on Education
mydata['EDU_CATEGORY']=pd.cut(mydata.EDUCATION ,[0,1,2,3,4],labels=["Graduate School","UNIVERSITY","HIGH SCHOOL","OTHERS"])

mydata.EDU_CATEGORY
mydata.isnull().sum()

#Grouping based on Marital Status
mydata['MAR_CATEGORY']=pd.cut(mydata.MARRIAGE,[0,1,2,3],labels=["Married","Single","OTHERS"])

#DEFAULT
#Creating Bar plot for DEFAULT
fig=sns.countplot(x='DEFAULT',data=mydata)
fig.set_xticklabels(["No Default","Default"])


#SEX

male_customers=(mydata.SEX==1).sum()
print(male_customers)

female_customers=(mydata.SEX==2).sum()
print(female_customers)

male_def=(mydata[mydata.SEX==1].DEFAULT==1).sum()
print(male_def)

female_def=(mydata[mydata.SEX==2].DEFAULT==1).sum()
print(female_def)

percent_male_def=round((male_def/male_customers)*100,2)
print(percent_male_def)

percent_female_def=round((female_def/female_customers)*100,2)
print(percent_female_def)

temp_mydata=pd.DataFrame({"Non-Defaulters":{"Male":100-percent_male_def,"Female":100-percent_female_def},"Defaulters":{"Male":percent_male_def,"Female":percent_female_def}})

fig=temp_mydata.plot(kind="bar")
fig.set_title("Percenatge of Male & Female Non -Defaulters vs Defaulters")
fig.set_ylabel("Percentage")

#EDUCATION
fig=sns.countplot(x='EDU_CATEGORY',data=mydata,hue='DEFAULT',palette="Set2")

#MARRIAGE
sns.countplot(x='MAR_CATEGORY',hue='DEFAULT',data=mydata,palette='Set3')


##BI-Variate Analysis

#correlation matrix
mydata1=mydata_raw.iloc[:,1:]
mydata1.columns

sns.set(rc={'figure.figsize':(25,8)})
sns.set_context("talk",font_scale=0.7)
sns.heatmap(mydata1.corr(),cmap='Greens', annot = True)

female_def=(mydata[mydata.SEX==2].DEFAULT==1).sum()

## Correlation between Repayment Status

mydata.PAY_1
var = ['PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
plt.figure(figsize=(8,8))
plt.title('Repayment Status')
corr=mydata[var].corr()
sns.heatmap(corr)
plt.show()


# AGE vs Default

mydata['AGE_CATEGORY']=pd.cut(mydata.AGE,[20,30,40,60],labels=["Young","Middle Aged","Senior Citizen"])
mydata.AGE_CATEGORY.head(10)
sns.countplot(x='EDUCATION',hue='DEFAULT',data=mydata)

##AGE DEFAULTER's
sns.violinplot(x="AGE_CATEGORY",y="LIMIT_BAL",data=mydata)

#Outlier detection
sns.catplot(x="DEFAULT",y="LIMIT_BAL",hue="DEFAULT",kind="boxen",data=mydata)

def boxplot_variation(feature1, feature2, feature3, width=16):
    fig, ax1 = plt.subplots(ncols=1, figsize=(width,6))
    s = sns.boxplot(ax = ax1, x=feature1, y=feature2, hue=feature3,
                data=mydata, palette="PRGn",showfliers=False)
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.show();
    
boxplot_variation('MARRIAGE','AGE', 'SEX',8)

boxplot_variation('EDUCATION','AGE', 'MARRIAGE',12)

### Spliting of Data into Train and Test
from sklearn.model_selection import train_test_split
x=mydata.drop('DEFAULT',axis=1)
y=mydata['DEFAULT']
xtrain,xtest,ytrain,ytest = train_test_split(x,y, test_size=0.3)

print("Number transactions train dataset: ", xtrain.shape)
print("Number transactions test dataset: ", xtest.shape)

xtrain
## SMOTE

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2)

x_train_res, y_train_res = sm.fit_sample(xtrain, ytrain.ravel())

print("Number transactions train dataset: ", x_train_res.shape)

## Scaling of the features
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train_res)
X_test=sc.fit_transform(xtest)

##RANDOM FOREST

# Running the Model
from sklearn.ensemble import RandomForestClassifier
classifier_RF= RandomForestClassifier(n_estimators=100,criterion ='gini')
classifier_RF.fit(X_train,ytrain)

# Predicting the results
y_pred=classifier_RF.predict(X_test)

# Model Performance
# Confusion Matrix
confusion_matrix = pd.crosstab(ytest, y_pred, rownames=['Actual'], colnames=['Predicted'], margins = True)
print(confusion_matrix)
sns.heatmap(confusion_matrix,annot=True,fmt='d',linewidths=.9)

#Gini Value
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(ytest, y_pred))

#AUC
y_pred_proba_RF = classifier_RF.predict_proba(X_test)[::,1]
fpr1, tpr1, _ = metrics.roc_curve(ytest,  y_pred_proba_RF)
auc1 = metrics.roc_auc_score(ytest, y_pred_proba_RF)
print(auc1)

# ROC
plt.figure(figsize=(10,7))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr1,tpr1,label="Random Forest, auc="+str(round(auc1,2)))

## Logistic Regression

#Running the Model
from sklearn.linear_model import LogisticRegression
classifier_LR = LogisticRegression(random_state=0)
classifier_LR.fit(X_train,ytrain)

# Predicting the results
y_pred=classifier_LR.predict(X_test)

# Making the Confusion Matrix
confusion_matrix = pd.crosstab(ytest, y_pred, rownames=['Actual'], colnames=['Predicted'], margins = True)
print(confusion_matrix)
sns.heatmap(confusion_matrix,annot=True,fmt='d',linewidths=.9)

# Model Performance

# Model Accuracy
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(ytest, y_pred))

# AUC
y_pred_proba_LR = classifier_LR.predict_proba(X_test)[::,1]
fpr2, tpr2, _ = metrics.roc_curve(ytest,  y_pred_proba_RF)
auc2 = metrics.roc_auc_score(ytest, y_pred_proba_RF)
print(auc2)

# ROC
plt.figure(figsize=(10,7))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr2,tpr2,label="Logistic Regression, auc="+str(round(auc2,2)))

## XGboost

#Running the Model
from xgboost import XGBClassifier
classifier_XGB=XGBClassifier()
classifier_XGB.fit(X_train,ytrain)

# Predicting the results
y_pred=classifier_XGB.predict(X_test)

#Making the Confusion Matrix
confusion_matrix = pd.crosstab(ytest, y_pred, rownames=['Actual'], colnames=['Predicted'], margins = True)
print(confusion_matrix)
sns.heatmap(confusion_matrix,annot=True,fmt='d',linewidths=.9)

#Model Performance

# Model Accuracy
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(ytest, y_pred))

#ROC
y_pred_proba_RF = classifier_XGB.predict_proba(X_test)[::,1]
fpr3, tpr3, _ = metrics.roc_curve(ytest,  y_pred_proba_RF)
auc3 = metrics.roc_auc_score(ytest, y_pred_proba_RF)
print(auc3)

# ROC
plt.figure(figsize=(10,7))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr3,tpr3,label="XGBoost, auc="+str(round(auc3,2)))

## Comparing the Models

## Plotting all the 3 models ROC
plt.figure(figsize=(10,7))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr1,tpr1,label="Random Forest, auc="+str(round(auc1,2)))
plt.plot(fpr2,tpr2,label="Logistic Regression, auc="+str(round(auc2,2)))
plt.plot(fpr3,tpr3,label="XGBoost, auc="+str(round(auc3,2)))
plt.legend(loc=4, title='Models', facecolor='white')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC', size=15)
plt.box(False)
plt.savefig('ImageName', format='png', dpi=200, transparent=True)