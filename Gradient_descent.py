# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 12:31:22 2022

@author: minje
"""

import numpy as np
import matplotlib.pyplot as plt
    

def hypothesis(theta, x):
    ret = 0
    
    # please implement the hypothesis equation shown in p16, W5-1.pdf
    ret = theta[0] + theta[1] * x
    
    return(ret)


# The cost function J(theta)
def cost_function(X, y, theta):
    ret = 0
    # n : Number of training examples
    n = len(X)
    
    # please import the cost function using hypothesis function above shown in p8, W5-2.pdf
    
    ret = np.sum((y - hypothesis(theta, X))**2)/n
    
    return(ret)

def comparison_cost(X, y, t1, t0):
    
    cost = cost_function(X, y, [t0, t1])
    
    # Evaluate the cost function as some values
    print("Cost for minimal parameters: ", cost, ", with theta0 =", t0, " and theta1 =", t1) # Best possible (got it from numpy)
    print("Cost for other theta: ", cost_function(X, y, [8.4, 0.6]))
    
    
    
       

def gradient(X, y, theta): 
    n = len(X)
    
    g = np.array([0,0])
    
    # 업데이트 하는데 사용할 가중치 값을 계산
    for j in range(n):
        g[1] += -(2 /n) * (X[j] * (y[j] - (X[j] * theta[1] + theta[0])))     # 비용함수를 기울기에 대해 편미분한 값  
        g[0] += -(2/n) * (y[j] - (X[j] * theta[1] + theta[0]))      # 비용함수를 절편에 대해 편미분한 값
        
        #implement the update rule distributed in p24, W5-2       
    
    return (g /np.linalg.norm(g))
    

      
        
def optimise_by_gradient_descent(X, y, t1, t0):
    # Start with some value for theta
    theta = np.array([0,0])     #theta[0] : 절편, theta[1] : 기울기
    
    
    list0 = np.array([])
    list1 = np.array([])
    
    
    # learning rate
    alpha = 0.2
    
    # number of steps
    steps = 100
    
    t_0 = 0.0 # 초기 절편
    t_1 = 0.0 # 초기 기울기 
    
    # gradient descent
    for s in range(steps):
        
        list0 = np.append(list0, t_0)      # save data for drawing
        list1 = np.append(list1, t_1)
        
        
        # parameter update part - Using gradient function, please implement the parameter update
        # as described in the algorithm in p21, W5-2.pdf
                
        t0_grad, t1_grad = gradient(X, y, [t_0, t_1])   # gradient함수를 통해 가중치 값을 받아옴
        
        
        t_0 = t_0 - alpha * t0_grad     # 절편 값을 업데이트
        t_1 = t_1 - alpha * t1_grad     # 기울기 값을 업데이트
        
        if s % 10 == 0:
            print([t_0, t_1])
        
    
        
    theta = [t_0, t_1]    
    print("Gradient descent gives after ", steps, "steps: ", theta)
    print("Best theta :", [t0, t1])
    
    
    
def main():
    
    # Training set
    Tx = np.array([2, 7, 13, 16, 22, 27, 35, 45, 50])
    Ty = np.array([5, 20, 14, 32, 22, 38, 39, 59, 70])
    
    # Draw the Training set
    plt.figure(figsize = (10, 8))
    plt.plot(Tx, Ty, 'X')
    plt.title("Training set", fontsize = 20)
    plt.xlabel("Weeks living in jeju", fontsize = 18)
    plt.ylabel("# of having black-pork ", fontsize = 18)
    
    # Best fit(by using the built-in function of numpy)
    # This is what we want to find by ourself in the following
    t1, t0 = np.polyfit(Tx, Ty, 1)
    plt.plot(Tx, t0 + t1*Tx)
    print("theta0 :", t0, "theta1 :", t1)
    plt.show()
    
    #HW part
    #t1, t0 is the fitted parameter which stands for the best possible (got it from numpy)
    comparison_cost(Tx, Ty, t1, t0)
    
    optimise_by_gradient_descent(Tx, Ty, t1, t0)
    
    
if __name__ == '__main__':
    main()