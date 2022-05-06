# -*- coding: utf-8 -*-
"""
Created on Wed May  4 10:57:07 2022

@author: Administrator
"""
import pickle
import scipy.stats as stat
import pylab
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import logging as lg
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
os.chdir("E:\Ivy-Professional-School\Python\ML\Activity Recognition System\Code")
import log_info
get_log=log_info.getLog()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

usable_directory=[]
#reading the folder
os.chdir("E:\Ivy-Professional-School\Python\ML\Activity Recognition System\Dataset")
for i in range(len(os.listdir())):
    try:
        if os.listdir()[i][-3:]!="pdf":
            usable_directory.append(os.listdir()[i])
            get_log.info(os.listdir()[i]+" folder has added in the directory list")
    except Exception as e:
        get_log.info(e+" error has occured during add directory name in a list")               

usable_directory

#reading all the data and merge to pandas dataframe
get_log.info("Creating a blank list for storing dataframes")
concat_dfs=[]
for folders in usable_directory:
    try:
        os.chdir("E:/Ivy-Professional-School/Python/ML/Activity Recognition System/Dataset"+"/"+folders)
        get_log.info("Entering in the folder named as "+ folders)
    except:
        get_log.info("Error has occured named as "+Exception)
    try:
        for files in os.listdir():
            get_log.info("Now inside in the folder we are seeing "+files)
            if files[-6:]=="ch.csv":
                df=pd.read_csv(files,error_bad_lines=False)
                df["Label"]=folders
                concat_dfs.append(df)
    except:
        get_log.info("Error has occured when it is trying to check files in folder, the error is "+Exception)
concat_dfs
#concatenating
get_log.info("Concatenate dataframes!")
df=pd.concat(concat_dfs)
#reading the file
df.head()
#dropping the first column
try:
    df.drop("# Columns: time",axis=1,inplace=True)
    get_log.info("Dropped # Columns: time column")
except Exception as e:
    get_log.info("Issue when trying to dropping the column error is "+e)

df.head()

#checking nulls

get_log.info("Checking for dtypes")

df.info()



#dropping nulls
get_log.info("Dropping nulls")
df=df[~df["avg_rss12"].isnull()]
df.isnull().sum()
#checking the distribution of the data
get_log.info("We are going to make distribution")
for i in df.columns:
    plt.hist(df[i])
    plt.title(f"Distribution of {i}")
    plt.show()


def plot_data(df,feature):
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    df[feature].hist()
    plt.title(f"Distribution of {feature}")
    plt.subplot(1,2,2)
    stat.probplot(df[feature],dist="norm",plot=pylab)
    plt.title(f"QQ Plot of {feature}")
    plt.show()




#we should check the var_rss13 column
get_log.info("Converting var_rss13 to log normal distribution")
#handiling var_rss13 column
df["var_rss13"]=df["var_rss13"].replace(0.00,df["var_rss13"].median())
df["var_rss13"]=np.log(df["var_rss13"])
plot_data(df, "var_rss13")
#handiling var_rss12 column
df["var_rss12"]=df["var_rss12"].replace(0.00,df["var_rss12"].median())
plot_data(df, "var_rss12")
df["var_rss12"]=np.log(df["var_rss12"])
#handiling var_rss23 column
df["var_rss23"]=df["var_rss23"].replace(0.00,df["var_rss23"].median())
plot_data(df, "var_rss23")
df["var_rss23"]=np.log(df["var_rss23"])

#encoding the label column
df["Label"].unique()

df["Label"]=df["Label"].replace("bending1",0)
df["Label"]=df["Label"].replace("bending2",1)       
df["Label"]=df["Label"].replace("cycling",2)
df["Label"]=df["Label"].replace("lying",3)
df["Label"]=df["Label"].replace("sitting",4)
df["Label"]=df["Label"].replace("standing",5)
df["Label"]=df["Label"].replace("walking",6)


df.head()


#checking multicolinearity
plt.figure(figsize=(10,6)) 
sns.heatmap(df[["avg_rss12","var_rss12","avg_rss13","var_rss13","avg_rss23","var_rss23"]].corr(),annot=True,cmap="Greens")
#checking dataset is imbalanced or not
df.groupby(by="Label")["Label"].count()





#train test splitting
get_log.info("Splitting the data")
df_train,df_test=train_test_split(df,train_size=0.80,random_state=100)
#y train and x train
get_log.info("Deviding the train data x and y")
y_train=df_train.pop("Label")
x_train=df_train
#y test and x test
get_log.info("Dividing the test data x and y")
y_test=df_test.pop("Label")
x_test=df_test

#standardisation
get_log.info("Scaling the data")
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)


#bulding the model
get_log.info("Using newton-cg solver")
lg_newton=LogisticRegression(solver="newton-cg",verbose=1,multi_class="ovr",class_weight='balanced')
#fit the data
get_log.info("Fit the model")
lg_newton.fit(x_train,y_train)
#doing the prediction
get_log.info("Doing prediction on training dataset")
y_train_pred=lg_newton.predict(x_train)
y_train_pred
#checking the probability of the prediction
y_train_prob_pred=lg_newton.predict_proba(x_train)







#confusion metrics
get_log.info("Create confusion matrices")
cnf_matrix=metrics.confusion_matrix(y_train, y_train_pred)
#checking the model
round(metrics.precision_score(y_train,y_train_pred,average="weighted"),2)

round(metrics.recall_score(y_train,y_train_pred,average="weighted"),2)

target_names=["0","1","2","3","4","5","6"]
print(classification_report(y_train,y_train_pred,target_names=target_names))

round(metrics.accuracy_score(y_train,y_train_pred),2)




y_train=np.array(y_train)
y_train_pred=np.array(y_train_pred)





from sklearn.metrics import roc_auc_score

def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

  #creating a set of all the unique classes using the actual class list
  unique_class = set(actual_class)
  roc_auc_dict = {}
  for per_class in unique_class:
    #creating a list of all the classes except the current class 
    other_class = [x for x in unique_class if x != per_class]

    #marking the current class as 1 and all other classes as 0
    new_actual_class = [0 if x in other_class else 1 for x in actual_class]
    new_pred_class = [0 if x in other_class else 1 for x in pred_class]

    #using the sklearn metrics method to calculate the roc_auc_score
    roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
    roc_auc_dict[per_class] = roc_auc

  return roc_auc_dict

print("\nLogistic Regression")
# assuming your already have a list of actual_class and predicted_class from the logistic regression classifier
lr_roc_auc_multiclass = roc_auc_score_multiclass(y_train, y_train_pred)
print(lr_roc_auc_multiclass)






lg_sag=LogisticRegression(solver="sag",verbose=1,multi_class="ovr",class_weight='balanced')
#fit the data
get_log.info("Fit the model")
lg_newton.fit(x_train,y_train)
#doing the prediction
get_log.info("Doing prediction on training dataset")
y_train_pred=lg_newton.predict(x_train)
y_train_pred
#checking the probability of the prediction
y_train_prob_pred=lg_newton.predict_proba(x_train)

#confusion metrics
get_log.info("Create confusion matrices")
cnf_matrix=metrics.confusion_matrix(y_train, y_train_pred)
#checking the model
round(metrics.precision_score(y_train,y_train_pred,average="weighted"),2)

round(metrics.recall_score(y_train,y_train_pred,average="weighted"),2)

target_names=["0","1","2","3","4","5","6"]
print(classification_report(y_train,y_train_pred,target_names=target_names))

round(metrics.accuracy_score(y_train,y_train_pred),2)
y_train=np.array(y_train)
y_train_pred=np.array(y_train_pred)

from sklearn.metrics import roc_auc_score

def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

  #creating a set of all the unique classes using the actual class list
  unique_class = set(actual_class)
  roc_auc_dict = {}
  for per_class in unique_class:
    #creating a list of all the classes except the current class 
    other_class = [x for x in unique_class if x != per_class]

    #marking the current class as 1 and all other classes as 0
    new_actual_class = [0 if x in other_class else 1 for x in actual_class]
    new_pred_class = [0 if x in other_class else 1 for x in pred_class]

    #using the sklearn metrics method to calculate the roc_auc_score
    roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
    roc_auc_dict[per_class] = roc_auc

  return roc_auc_dict

print("\nLogistic Regression")
# assuming your already have a list of actual_class and predicted_class from the logistic regression classifier
lr_roc_auc_multiclass = roc_auc_score_multiclass(y_train, y_train_pred)
print(lr_roc_auc_multiclass)







lg_saga=LogisticRegression(penalty="l1",solver="saga",verbose=1,multi_class="ovr",class_weight='balanced')
#fit the data
get_log.info("Fit the model")
lg_newton.fit(x_train,y_train)
#doing the prediction
get_log.info("Doing prediction on training dataset")
y_train_pred=lg_newton.predict(x_train)
y_train_pred
#checking the probability of the prediction
y_train_prob_pred=lg_newton.predict_proba(x_train)

#confusion metrics
get_log.info("Create confusion matrices")
cnf_matrix=metrics.confusion_matrix(y_train, y_train_pred)
cnf_matrix

FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

tpr=TP/TP+FN
fpr=FP/FP+TN


#checking the model
round(metrics.precision_score(y_train,y_train_pred,average="weighted"),2)

round(metrics.recall_score(y_train,y_train_pred,average="weighted"),2)

round(metrics.accuracy_score(y_train, y_train_pred),2)
round(metrics.f1_score(y_train, y_train_pred,average="weighted"),2)

target_names=["0","1","2","3","4","5","6"]
print(classification_report(y_train,y_train_pred,target_names=target_names))

round(metrics.accuracy_score(y_train,y_train_pred),2)
y_train=np.array(y_train)
y_train_pred=np.array(y_train_pred)

from sklearn.metrics import roc_auc_score

def roc_auc_score_multiclass(actual_class, pred_class, average = "weighted"):

  #creating a set of all the unique classes using the actual class list
  unique_class = set(actual_class)
  roc_auc_dict = {}
  for per_class in unique_class:
    #creating a list of all the classes except the current class 
    other_class = [x for x in unique_class if x != per_class]

    #marking the current class as 1 and all other classes as 0
    new_actual_class = [0 if x in other_class else 1 for x in actual_class]
    new_pred_class = [0 if x in other_class else 1 for x in pred_class]

    #using the sklearn metrics method to calculate the roc_auc_score
    roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
    roc_auc_dict[per_class] = roc_auc

  return roc_auc_dict

print("\nLogistic Regression")
# assuming your already have a list of actual_class and predicted_class from the logistic regression classifier
lr_roc_auc_multiclass = roc_auc_score_multiclass(y_train, y_train_pred)
print(lr_roc_auc_multiclass)

lr_roc_auc_multiclass


#use the test data


x_test=scaler.transform(x_test)

y_test

y_test_pred=lg_newton.predict(x_test)
prob=lg_newton.predict_proba(x_test)
get_log.info("Create confusion matrices for test data solver newton")
cnf_matrix=metrics.confusion_matrix(y_test, y_test_pred)
cnf_matrix

FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)


metrics.accuracy_score(y_test, y_test_pred)

pickle.dump(lg_newton,open("activity.pickle",'wb'))

model=pickle.load(open("activity.pickle",'rb'))
def pred(avg_rss12,var_rss12,avg_rss13,var_rss13,avg_rss23,var_rss23):
    #transformed data
    get_log.info("Data has been taken by the model and transforming")
    pred=scaler.transform([[avg_rss12,var_rss12,avg_rss13,var_rss13,avg_rss23,var_rss23]])
    try:
        prediction=model.predict(pred)
        get_log.info("Prediction Done!!")
        if prediction==0:
            return "Bending 1"
        elif prediction==1:
            return "Bending 2"
        elif prediction==2:
            return "Cycling"
        elif prediction==3:
            return "Lying"
        elif prediction==4:
            return "Sitting"
        elif prediction==5:
            return "Standing"
        return "Walking"
    except:
        return "There is a issue with the source file"





# metrics.f1_score(y_test, y_test_pred,average="weighted")
# roc_auc_score_multiclass(y_test,y_test_pred)

# pickle.dump(lg_newton,open("activity.pickle",'wb'))

# model=pickle.load(open("activity.pickle",'rb'))

# t=scaler.transform([[23.75,0.41,24,0.8,29,0.47]])

# pred=model.predict([[-0.71207862,  1.8682965 ,  0.49072785,  1.08586715,  0.68918038,
#        -0.38444627]])
# pred





