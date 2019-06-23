# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 20:55:41 2019

@author: Arnold Yu
@discription: This is a recommendation system using matrix factorization techniques.
"""

import pandas as pd
import numpy as np
from sortedcontainers import SortedList
import copy
import itertools
from numpy.linalg import norm
from sklearn.decomposition import NMF

# row is user, column is business
matrix = pd.read_csv('user_business_preprocess.csv',  index_col = 0).fillna(0)

# store columns and rows name
columns_name = list(matrix.columns.values)
rows_name = list(matrix.index)

mu = np.mean(list(matrix.values))

# matrix factorization with regularization
alpha_1 = 0.01
l1_l2_ratio = 0.05
model = NMF(n_components=10, solver = 'cd',tol = 0.0001 , max_iter = 100000, alpha = alpha_1, l1_ratio  = l1_l2_ratio, init='random', random_state=0)
W = model.fit_transform(matrix)
H = model.components_
"""
R_hat = matrix.values - W.dot(H) + alpha_1 * l1_l2_ratio * norm(W, ord = 1) + alpha_1 * l1_l2_ratio *norm(H, ord = 1) + alpha_1 *(1-l1_l2_ratio)/2*norm(W, ord = 2) + \
        alpha_1 *(1-l1_l2_ratio)/2*norm(H, ord = 2)
"""

prediction = W.dot(H) + alpha_1 * l1_l2_ratio * norm(W, ord = 1) + alpha_1 * l1_l2_ratio *norm(H, ord = 1) + alpha_1 *(1-l1_l2_ratio)/2*norm(W, ord = 2) + \
        alpha_1 *(1-l1_l2_ratio)/2*norm(H, ord = 2) + mu

# root of mean square error
rmse = np.mean(list((matrix.values-W.dot(H))**2)) ** 0.5
# 0.22119463845978807
print("RMSE of DataSet : %f"  %rmse)


search = copy.deepcopy(prediction)
search = pd.DataFrame(search)
search.columns = columns_name

business = pd.read_csv('business_preprocess_filtered.csv')
def calculateReleventR(dict_, N):

    sorted_x = sorted(dict_.items(), key=lambda kv: kv[1], reverse = True)
    if len(sorted_x) >= N:
        a11 = sorted_x[:N]
    else:
        a11 = sorted_x
    list_ = {}
    for i in range(0,len(a11)):
        list_[a11[i][0]] = i
                
    result = [0] * N
    for row in business.iterrows():
        if row[1]['business_id'] in list_:
            if row[1]['stars'] >= 3.5:
                result[list_[row[1]['business_id']]] = 1
    return result

def averagePercision( list_):
    m = sum(list_)
    if m == 0:
        return 0
    k = len(list_)
    ap = 0.0
    for i in range(0,k):
        ap += float(sum(list_[0:i+1]))/float(len(list_[0:i+1]))*list_[i]
    return ap/float(m)
        
def averageRecall(list_):
        
    m = sum(list_)
    if m == 0:
        return 0
    k = len(list_)
    ar = 0.0
    for i in range(0,k):
        ar += float(sum(list_[0:i+1]))/float(m)*list_[i]
            
    return ar/float(m)

sum_ap = 0
sum_ar = 0

dict_ = {}
for i in range(0,matrix.shape[0]):
    dict_[i] = {}
    for col in range(0,matrix.shape[1]):
        dict_[i][columns_name[col]] = search[columns_name[col]][i]
    
    
for k, v in dict_.items():
    result = calculateReleventR(v,10)
    ap = averagePercision(result)
    ar = averageRecall(result)
    sum_ap += ap
    sum_ar += ar

# sum_ap = 15713.6701648719
# sum_ar = 8867.596825397477
meanap = sum_ap / float(search.shape[0]) #0.9816135785152362
meanar = sum_ar / float(search.shape[0]) #0.5539478276735056

print("MAP@10: %f" %meanap)
print("MAR@10: %f" %meanar)

# predict user '-318sKiQDgbjLzF4FCU1XA'
for i in range(0, matrix.shape[0]):    
    if matrix.iloc[i].name == "'-318sKiQDgbjLzF4FCU1XA'":
        row = matrix.iloc[i]
        rownumber = i
    
predicted_row = prediction[rownumber]


for i in range(0, row.shape[0]):
    row[i] = predicted_row[i]

s = pd.DataFrame(row.sort_values(ascending=False))

dict_ = {}
for i in range(0,10):
    dict_[i+1] = s.iloc[i].name
    

            
        
sorted_x = sorted(dict_.items(), key=lambda kv: kv[1], reverse = True)
if len(sorted_x) >= 10:
    a11 = sorted_x[:10]
else:
    a11 = sorted_x
# store the top 10 buisness_id
list_ = {}
for i in range(0,len(a11)):
    list_[a11[i][1]] = a11[i][0]
            
for row in business.iterrows():
    if row[1]['business_id'] in list_:
        list_[row[1]['business_id']] = (list_[row[1]['business_id']] ,row[1]['name'] + row[1]['address'] + row[1]['city']+ row[1]['state'] + row[1]['postal_code'])              
               
dict4 = {}
for k,v in list_.items():
    dict4[v[0]] = v[1]
for k in sorted(dict4.keys()):
    print(k, dict4[k])
    
    
