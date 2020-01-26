
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

#checking for null values
mydata.isnull().sum()
plt.plot(mydata.isnull().sum())

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

# Dropping ID Column
mydata=mydata.drop('ID',axis=1)
### Spliting of Data into Train and Test
from sklearn.model_selection import train_test_split
x=mydata.drop('DEFAULT',axis=1)
y=mydata['DEFAULT']
xtrain,xtest,ytrain,ytest = train_test_split(x,y, test_size=0.3)

print("Number transactions train dataset: ", xtrain.shape)
print("Number transactions test dataset: ", xtest.shape)

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
classifier_RF.fit(X_train,y_train_res)

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

## Feature selection
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_)

#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

## Making a data set with only the important features

## Logistic Regression

#Running the Model
from sklearn.linear_model import LogisticRegression
classifier_LR = LogisticRegression(random_state=0)
classifier_LR.fit(X_train,y_train_res)

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
fpr2, tpr2, _ = metrics.roc_curve(ytest,  y_pred_proba_LR)
auc2 = metrics.roc_auc_score(ytest, y_pred_proba_LR)
print(auc2)

# ROC
plt.figure(figsize=(10,7))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr2,tpr2,label="Logistic Regression, auc2="+str(round(auc2,1)))

## XGboost

#Running the Model
from xgboost import XGBClassifier
classifier_XGB=XGBClassifier()
classifier_XGB.fit(X_train,y_train_res)

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
y_pred_proba_XGB = classifier_XGB.predict_proba(X_test)[::,1]
fpr3, tpr3, _ = metrics.roc_curve(ytest,  y_pred_proba_XGB)
auc3 = metrics.roc_auc_score(ytest, y_pred_proba_XGB)
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