# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 18:36:17 2023

@author: minje
"""
# 은닉층과 출력층의 활성화 함수로 모두 시그모이드를 사용, 가장 기본 파일
# Scikit-learn에서 제공하는 MLP를 이용해 구현한 부분을 포함! 
# (다른 MLP_iris 파일들에는 사이킷런 이용해 구현한 부분 없습니다.)

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Iris dataset 불러오기
from sklearn.datasets import load_iris
iris = load_iris()

# check the target features of the dataset
print (iris['target_names'])  # ['setosa' 'versicolor' 'virginica']
print (iris['feature_names']) # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# check the shape of descriptive features of the sample
print (iris['data'].shape)  # (150, 4)


X = iris.data[:, 0:4]  
Y = iris.target.reshape(-1)

# One-hot encoding 형태로 
y_one_hot = pd.get_dummies(Y)


# Train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.4, random_state=0)

#-----------------------------------------------------------------------------------------#
# 우선, Scikit-learn에서 제공하는 MLPClassifier로 구현해보자 
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(max_iter=1000, random_state = 0)
mlp.fit(x_train,y_train)

y_test_pred = mlp.predict(x_test)
scikit_accuracy = accuracy_score(y_test, y_test_pred)
print('scikit Accuracy: %.2f' % scikit_accuracy)

#-------------------------------------------------------------------------------------------#
# 직접 구현
# 작년에 내주신 기계학습 9주차 과제의 Perceptron.py 파일을 참고하여 코드 작성했습니다. 
from scipy.special import expit

class MLP:
    
    def __init__(self, input_size, hidden_size, output_size, lr):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr

         
        np.random.seed(seed = 0)
        
        # 가중치를 랜덤하게 초기화시킨다.
        # 입력층 ~ 은닉층 사이에 존재하는 가중치, weights1
        self.weights1 = np.random.randn(self.input_size, self.hidden_size) 
        # * np.sqrt(2.0/(input_size + hidden_size))
        # print(self.weights1.shape) # (4, hidden_size)
        
        # 은닉층 ~ 출력층 사이에 존재하는 가중치, weights2
        self.weights2 = np.random.randn(self.hidden_size, self.output_size) 
        # * np.sqrt(2.0/(hidden_size + output_size))
        # print(self.weights2.shape) # (hidden_size, 3)
        
    
    def sigmoid(self, x):
# =============================================================================
#         if (np.any(x) >= 0):
#             return 1 / (1 + np.exp(-x))
#         else:
#             return np.exp(x) / (1 + np.exp(x))
# =============================================================================
         return expit(x)
    
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    

    # 순전파(전방계산)
    def feedforward(self, X):
        
        self.Zsumj = np.dot(X, self.weights1)
        self.Zj = self.sigmoid(self.Zsumj)
        
        self.Osumk = np.dot(self.Zj, self.weights2)
        self.Ok = self.sigmoid(self.Osumk)
        
        return self.Ok
        
    
    # 역전파
    def backpropagation(self, X, y, output):
        
        # m = X.shape[0] # 전체 데이터의 개수
        
        self.delta_k = (y - output) * self.sigmoid_derivative(self.Osumk) 
        self.weights2 -= (-1) * self.Zj.T.dot(self.delta_k) * self.lr  
        # / m  # 평균 그레이디언트로 갱신
        
        self.eta_j = self.delta_k.dot(self.weights2.T) * self.sigmoid_derivative(self.Zsumj) 
        self.weights1 -= (-1) * X.T.dot(self.eta_j) * self.lr  
        # / m  # 평균 그레이디언트로 갱신
        
# =============================================================================
#         self.delta_k = (y - output) * self.sigmoid_derivative(self.Osumk) 
#         self.weights2 -= (-1) * self.Zj.T.dot(self.delta_k) * self.lr
#         
#         self.eta_j = self.delta_k.dot(self.weights2.T) * self.sigmoid_derivative(self.Zsumj)
#         self.weights1 -= (-1) * X.T.dot(self.eta_j) * self.lr
# =============================================================================
        
    def train(self, X, y, epochs):
        
        for epoch in range(epochs):
            output = self.feedforward(X)
            # print("output print 시작")
            # print(output)
            
            # loss 계산
            # loss = (0.5 * (output - y)**2).sum()
            # loss_arr.append(loss)
            
            self.backpropagation(X, y, output)
            
            
    def train_mini_batch(self, X, y, epochs, batch_size):
        
        m = X.shape[0]  # 전체 데이터의 개수

        for epoch in range(epochs):
            
        # 미니배치 경사하강법을 위해 데이터를 랜덤하게 섞기
            indices = np.random.permutation(m) # 배열의 인덱스를 무작위로 섞음
            X_shuffled = X[indices]
            y_shuffled = y[indices]

        # 미니배치 사이즈 단위로 학습을 진행
            for i in range(0, m, batch_size): 
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                output = self.feedforward(X_batch)
                self.backpropagation(X_batch, y_batch, output)
            
            
    def predict(self, X):
        
        # 항상 3개 열로 pred_one_hot이 반환될 수 있도록, 코드를 수정해 주었다.
        pred = self.feedforward(X)
        # print("pred print 시작")
        # print(pred.shape)
        # print(pred)
        
        pred_one_hot = np.zeros_like(pred)
        pred_one_hot[np.arange(len(pred)), pred.argmax(axis=1)] = 1
        return pred_one_hot
    
    
    def print_loss(self, loss_arr):
        
        import matplotlib.pyplot as plt
        plt.plot(loss_arr)



# MLP training and testing
mlp = MLP(input_size = 4, hidden_size = 15, output_size = 3, lr = 0.01)
# loss_arr = []
mlp.train(x_train, y_train, epochs=1000)
# mlp.print_loss(loss_arr)

# test set에서의 정확도
y_pred = mlp.predict(x_test)
# print(y_pred)  
accuracy = accuracy_score(y_pred, y_test)
print("class MLP - default Accuracy: %.2f " % accuracy)