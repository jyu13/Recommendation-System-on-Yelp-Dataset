#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 17:49:41 2019

@author: Arnold
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from keras.layers import Dropout, BatchNormalization, Activation
from keras.regularizers import l2
from keras.optimizers import SGD, Adam

# load in the data
data = pd.read_csv('user_business_preprocess.csv',  index_col = 0)

row_no = sum(data.count())
N = data.shape[0] # number of users
M = data.shape[1] # number of busniesses

# store columns and rows name
columns_name = list(data.columns.values)
rows_name = list(data.index)

# build column_no by 3 matrix 
matrix = np.empty((row_no, 3), dtype= float)

data = data.fillna(0)
k = 0
#data.iloc[0].name 
#data.columns[1]
for i in range(0,data.shape[0]):
    for j in range(0, data.shape[1]):
        if data[data.columns[j]][data.iloc[i].name] != 0:
            matrix[k][0] = rows_name.index(data.iloc[i].name) + 1
            matrix[k][1] = columns_name.index(data.columns[j]) + 1
            matrix[k][2] = data[data.columns[j]][data.iloc[i].name]
            k += 1
            
matrix = pd.DataFrame(matrix)
matrix.columns = ['User', 'Business', 'Rating']
"""
matrix.to_csv('user_business_rating_matrix.csv', sep = ",")
data_nn = pd.read_csv('user_business_rating_matrix.csv', sep = ",", index_col = 0)
"""
# split into train and test
data_nn = shuffle(matrix)
cutoff = int(0.8*len(data_nn))
data_train = data_nn.iloc[:cutoff]
data_test = data_nn.iloc[cutoff:]

data_test.User.values

# initialize variables
K = 10 # latent dimensionality
mu = data_train.Rating.mean()
epochs = 15
reg = 0.0001 # regularization penalty


# keras model
u = Input(shape=(1,))
m = Input(shape=(1,))
u_embedding = Embedding(N, K, embeddings_regularizer = l2(reg))(u) # (N, 1, K)
m_embedding = Embedding(M, K, embeddings_regularizer = l2(reg))(m) # (N, 1, K)
u_embedding = Flatten()(u_embedding) # (N, K)
m_embedding = Flatten()(m_embedding) # (N, K)
x = Concatenate()([u_embedding, m_embedding]) # (N, 2K)

# the neural network
x = Dense(400)(x)
# x = BatchNormalization()(x)
x = Activation('relu')(x)
# x = Dropout(0.5)(x)
# x = Dense(100)(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
x = Dense(1)(x)

model = Model(inputs=[u, m], outputs=x)
model.compile(
  loss='mse',
  # optimizer='adam',
  # optimizer=Adam(lr=0.01),
  optimizer=SGD(lr=0.08, momentum=0.9),
  metrics=['mse'],
)

r = model.fit(
  x=[data_train.User.values, data_train.Business.values],
  y=data_train.Rating.values - mu,
  epochs=epochs,
  batch_size=128,
  validation_data=(
    [data_test.User.values, data_test.Business.values],
    data_test.Rating.values - mu
  )
)

r.history

# plot losses
plt.plot(r.history['loss'], label="train loss")
plt.plot(r.history['val_loss'], label="test loss")
plt.legend()
plt.show()

# plot mse
plt.plot(r.history['mean_squared_error'], label="train mse")
plt.plot(r.history['val_mean_squared_error'], label="test mse")
plt.legend()
plt.show()


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

# build user-business dictionary


dict_ = {}
for k in range(0,N):
    dict_[k] = {}
    u_b = np.zeros((M,2),dtype = int)
    for i in range(0, M):
        u_b[i][0] = k+1
        u_b[i][1] = i+1
    u_b = pd.DataFrame(u_b)
    u_b.columns = ['User', 'Business']
    predict_ = model.predict(x = [u_b.User.values, u_b.Business.values])
    predict_ = pd.DataFrame(predict_)
    predict_.columns = ['Rating']
    final_result = pd.concat([u_b,predict_], axis = 1)
    for i in range(0,M):
        dict_[k][columns_name[final_result['Business'][i]-1]] = final_result['Rating'][i] + mu

sum_ap = 0
sum_ar = 0
    
for k, v in dict_.items():
    result = calculateReleventR(v,10)
    ap = averagePercision(result)
    ar = averageRecall(result)
    sum_ap += ap
    sum_ar += ar

# sum_ap = 15827.652876984112
# sum_ar = 8690.000000001228
meanap = sum_ap / float(data.shape[0]) #0.9887339378425857
meanar = sum_ar / float(data.shape[0]) #0.54285357321347

print("MAP@10: %f" %meanap)
print("MAR@10: %f" %meanar)







# predict "'-318sKiQDgbjLzF4FCU1XA'"
index_user = rows_name.index("'-318sKiQDgbjLzF4FCU1XA'")
u_b_array = np.zeros((M,2),dtype = int)
for i in range(0,M):
    u_b_array[i][0] = index_user+1
    u_b_array[i][1] = i+1
u_b_array = pd.DataFrame(u_b_array)
u_b_array.columns = ['User', 'Business']
predict_y = model.predict(x = [u_b_array.User.values, u_b_array.Business.values])
predict_y = pd.DataFrame(predict_y)
predict_y.columns = ['Rating']
final_result = pd.concat([u_b_array,predict_y], axis = 1)
final_result = final_result.sort_values(by = ['Rating'], ascending=False)

# Store business id 
dict_ = {}
for i in range(0,10):
    dict_[i+1] = columns_name[final_result.iloc[i].name]

# print the top 10 results    
                   
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
    


