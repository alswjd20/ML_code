# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 17:24:30 2022

@author: minje
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial import Voronoi, voronoi_plot_2d


"[speed, agility]"
data = [[2.50, 6.00], [3.75, 8.00],[2.25, 5.50], [3.25, 8.25],
        [2.75, 7.50], [4.50, 5.00],[3.50, 5.25], [3.00, 3.25],
        [4.00, 4.00], [4.25, 3.75],[2.00, 2.00], [5.00, 2.50],
        [8.25, 8.50], [5.75, 8.75],[4.75, 6.25], [5.50, 6.75],
        [5.25, 9.50], [7.00, 4.25],[7.50, 8.00], [7.25, 5.75]]


tf_draft = ['No', 'No', 'No', 'No', 'No', 'No',
            'No','No', 'No', 'No', 'No', 'No',
            'No', 'Yes', 'Yes', 'Yes', 'Yes',
            'Yes', 'Yes', 'Yes']


vor = Voronoi(data)
fig = voronoi_plot_2d(vor)
fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',
                line_width=2, line_alpha=0.6, point_size=2)
plt.show()


"1. 3-nearest neighbor"
"(2) Manhattan distance 구현"
manhatt_dist = list()

def manhattan_distance(data, new_data):
    manhatt_dist.clear()
    for i in range(0, len(data)):
        manhatt_distance = 0.0
        manhatt_distance += abs(data[i][0] - new_data[0]) 
        manhatt_distance += abs(data[i][1] - new_data[1])
        manhatt_dist.append([i, manhatt_distance])
    
    return manhatt_dist


"3NN-Manhattan"
nearest_3_manhatt = list()

def KNN3_Manhattan_Classify(data, tf_draft, new_data):
    nearest_3_manhatt.clear()
    manhattan_distance(data, new_data)      #거리 계산
    manhatt_dist.sort(key = lambda x:x[1])   #거리순으로 정렬
    nearest_3_manhatt.append(manhatt_dist[0])
    nearest_3_manhatt.append(manhatt_dist[1])
    nearest_3_manhatt.append(manhatt_dist[2]) # 가장 가까운 3개의 data 리스트 추출
    print(nearest_3_manhatt)
    
    
    cnt_no = 0
    cnt_yes = 0
 
    for i in range(0,3):        #가장 가까운 3개 data의 라벨을 확인
        if(tf_draft[nearest_3_manhatt[i][0]] == 'Yes'):     #라벨이 Yes에 해당되면
            cnt_yes += 1
        else:               #라벨이 Yes가 아니면, 즉, No에 해당되면
            cnt_no += 1
 
    
    # Yes의 개수와 No의 개수를 비교
    if(cnt_yes > cnt_no):
        print("해당 Data는 Yes로 판별되었습니다")
        data.append(new_data)           #새로운 data를 기존의 data에 추가
        tf_draft.append('Yes')          #판별된 라벨값(Yes)도 라벨 리스트에 추가
        
    else:
        print("해당 Data는 No로 판별되었습니다")
        data.append(new_data)       
        tf_draft.append('No')          #판별된 라벨값(No)도 라벨 리스트에 추가
        
        
        
# 다음 5개의 Data에 대해 어떻게 판별하는지 확인       
  
KNN3_Manhattan_Classify(data, tf_draft, [6.75, 3])
KNN3_Manhattan_Classify(data, tf_draft, [5.34, 6.0])
KNN3_Manhattan_Classify(data, tf_draft, [4.67, 8.4])
KNN3_Manhattan_Classify(data, tf_draft, [7.0, 7.0])
KNN3_Manhattan_Classify(data, tf_draft, [7.8, 5.4])

vor = Voronoi(data)
fig = voronoi_plot_2d(vor)
fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',
                line_width=2, line_alpha=0.6, point_size=2)
plt.show()



