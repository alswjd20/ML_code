# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 21:54:08 2023

@author: minje
"""

# default 파일과 다르게, 출력층의 활성화 함수로 소프트맥스 함수를 사용
# (defautlt 파일이랑 softmax 구현, feedforward, Backpropagation 함수 구현 부분만 다름)


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr

        # Initialize weights randomly
        np.random.seed(seed = 0)
        self.weights1 = np.random.randn(self.input_size, self.hidden_size) 
        # * np.sqrt(2.0/(input_size + hidden_size))
        self.weights2 = np.random.randn(self.hidden_size, self.output_size) 
        # * np.sqrt(2.0/(hidden_size + output_size))
        
    
    def sigmoid(self, x):
        
        from scipy.special import expit
# =============================================================================
#          if (np.any(x) >= 0):
#              return 1 / (1 + np.exp(-x))
#          else:
#             return np.exp(x) / (1 + np.exp(x))
# =============================================================================

        # return 1 / (1 + np.exp(-x))
        return expit(x)


    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    
    def softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)  # overflow 방지를 위해 최댓값을 빼줌
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    
    
    # 순전파(전방계산)
    def feedforward(self, X):
        
        self.Zsumj = np.dot(X, self.weights1)
        self.Zj = self.sigmoid(self.Zsumj)
        
        
        self.Osumk = np.dot(self.Zj, self.weights2)
        # print(self.Osumk)
        self.Ok = self.softmax(self.Osumk)
        #print(self.Ok)
        
        return self.Ok
        
    
    # 역전파
    def backpropagation(self, X, y, output):
        
        m = X.shape[0] # 전체 데이터의 개수
        
        # 출력층에서의 오차(그레이디언트), 가중치 업데이트
        error2 = output - y
        grad2 = self.Zj.T.dot(error2) / m  # 평균 그레이디언트로 갱신될 수 있도록
        self.weights2 -= grad2 * self.lr
        
        # 은닉층에서의 그레이디언트, 가중치 업데이트
        error1 = np.dot(error2, self.weights2.T) * self.sigmoid_derivative(self.Zsumj)
        grad1 = X.T.dot(error1) / m # 평균 그레이디언트로 갱신될 수 있도록
        self.weights1 -= grad1 * self.lr 
        
# =============================================================================
#         # 출력층에서의 오차(그레이디언트), 가중치 업데이트
#         error2 = output - y
#         grad2 = self.Zj.T.dot(error2) 
#         self.weights2 -= grad2 * self.lr
#          
#         # 은닉층에서의 그레이디언트, 가중치 업데이트
#         error1 = np.dot(error2, self.weights2.T) * self.sigmoid_derivative(self.Zsumj)
#         grad1 = X.T.dot(error1) 
#         self.weights1 -= grad1 * self.lr
# =============================================================================
        

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.feedforward(X)
            # print("output print 시작")
            # print(output)
            
            # loss 계산
            # loss = -np.sum(y * np.log(output))
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
        pred = self.feedforward(X)
        # print("pred 출력 시작")
        # print(pred)
        pred_one_hot = np.zeros_like(pred)
        pred_one_hot[np.arange(len(pred)), pred.argmax(axis=1)] = 1
        
        return pred_one_hot
    
    def print_loss(self, loss_arr):
        
        import matplotlib.pyplot as plt
        plt.plot(loss_arr)


# Iris dataset 불러오기
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

# One-hot encoding 형태로
y_one_hot = pd.get_dummies(y)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.4)

# MLP training and testing
mlp = MLP(input_size = 4, hidden_size = 60, output_size=3, lr = 0.01)
# loss_arr = []
mlp.train(X_train, y_train, epochs=1000)
# mlp.print_loss(loss_arr)

# test set에서의 정확도 
y_pred = mlp.predict(X_test)
# print("y_pred 출력 시작")
# print(y_pred)
accuracy = accuracy_score(y_pred, y_test)
print("class MLP - softmax Accuracy: %.2f " % accuracy)