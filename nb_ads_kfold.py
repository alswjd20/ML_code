# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


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
    