# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 15:06:09 2023

@author: minje
"""


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# 앞서 2개의 MLP_iris 파일에서 각각 만든 feedforward, Backpropagation을 통해 MNIST에서의 성능 확인


# MNIST 데이터셋 불러오기 
import tensorflow as tf
mnist = tf.keras.datasets.mnist
train_data, test_data = mnist.load_data()

X_train, y_train = train_data
X_test, y_test = test_data


# shape 학인
#print("X_train의 shape : ", X_train.shape) # (60000, 28, 28)
#print("y_train의 shape : ", y_train.shape) # (60000,)

#print("X_test의 shape :", X_test.shape) # (10000, 28, 28)
#print("y_test의 shape :", y_test.shape) # (10000,)


# 입력 데이터(X) 전처리, reshape 해서 2차원으로 만든 뒤, 실수로 만들고, 
# 픽셀의 최댓값인 255.0을 나누어 0~1 사이의 값으로 정규화
X_train = X_train.reshape(60000, 784).astype('float32') / 255.0
X_test = X_test.reshape(10000, 784).astype('float32') / 255.0



# 출력 레이블(Y) 전처리, 원핫인코딩 형태로 변환 
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)


print('y_train의 shape :', np.shape(y_train))
print('샘플 하나만 출력해보자, y_train의 열 번째 데이터 :', y_train[9])

# 다시 한 번 shape 학인
#print("X_train의 shape : ", X_train.shape) # (60000, 784)
#print("y_train의 shape : ", y_train.shape) # (60000, 10)

#print("X_test의 shape :", X_test.shape) # (10000, 784)
#print("y_test의 shape :", y_test.shape) # (10000, 10)


# =============================================================================
# MNIST에서도 Scikit-learn에서 제공하는 MLPClassifier 성능 확인 -> 0.96
# from sklearn.neural_network import MLPClassifier

# mlp = MLPClassifier(max_iter=1000, random_state = 0)
# mlp.fit(X_train,y_train)
# 
# y_test_pred = mlp.predict(X_test)
# scikit_accuracy = accuracy_score(y_test, y_test_pred)
# print('MNIST scikit Accuracy : %.2f' % scikit_accuracy)

# =============================================================================
#  내가 만든 MLP_iris_default 파일, MLP_iris_softmax 파일의 MLP 클래스로 성능 확인

import MLP_iris_softmax as mlp
# import MLP_iris_default as mlp

mnist_mlp = mlp.MLP(input_size = 784, hidden_size = 700, output_size = 10, lr = 1.5)
# mnist_mlp.train(X_train, y_train, epochs=100)
mnist_mlp.train_mini_batch(X_train, y_train, epochs=100, batch_size = 32) # 미니 배치 경사하강법으로 학습시키기
print("학습 완료")
 
# test set에서의 정확도
y_pred = mnist_mlp.predict(X_test)
print(y_pred)  
accuracy = accuracy_score(y_pred, y_test)
print("class MLP 이용 - MNIST softmax Accuracy: %.2f" % accuracy)
# print("class MLP 이용 - MNIST default Accuracy: %.2f" % accuracy)
