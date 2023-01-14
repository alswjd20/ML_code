# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 12:05:37 2021

@author: jeehang

acknowledgement: https://www.analyticsvidhya.com/blog/2021/01/a-guide-to-the-naive-bayes-algorithm/
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')


X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values


print(X)
print(X.shape)
print(".....")
print(y)
print(y.shape)


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

print("split 완료")



# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("feature scaling 완료")

# Training the Naive Bayes model on the Training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

print("training 완료")

# Predicting the Test set results
y_pred = classifier.predict(X_test)
ac = accuracy_score(y_test,y_pred)      # 정확도
print('accuracy(정확도): ', ac)

print('Precision(정밀도): %f' % precision_score(y_test, y_pred)) # 정밀도
print('Recall(재현율) : %f' % recall_score(y_test, y_pred))   # 재현율



# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[1,0])
print(cm)
print("혼동행렬 출력 완료")

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print('tp, fn, fp, tn = ', tp, fn, fp, tn)
sensitivity = tp/(tp+fn)    # 민감도
print('Sensitivity(민감도): ', sensitivity)
specificity = tn/(tn+fp)    # 특이도 
print('Specificity(특이도): ', specificity)



lr_probs = classifier.predict_proba(X_test)
lr_auc = roc_auc_score(y_test, lr_probs[:, 1])
print(lr_auc)

lr_auc = roc_auc_score(y_test, y_pred)
print(lr_auc)

# calculate roc curves
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs[:, 1])

# plot the roc curve for the model
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.show()


kfold = KFold(n_splits = 10, shuffle = False)
kf_accuracy = []



for train_index, test_index in kfold.split(X):
    # split 갯수(10)만큼 train, test set 분할해서 추출
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("Training Set index:", train_index)
    print("TEST Set index:", test_index)
    
    
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Training the Naive Bayes model on the Training set
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    ac = accuracy_score(y_test,y_pred)      # 정확도
    kf_accuracy.append(ac)
    
    print()
    

print("정확도 기록(performance 1 ~ 10) : ", kf_accuracy)
print("예측 정확도 평균 : ", np.mean(kf_accuracy))
print("분산 : ", np.var(kf_accuracy))