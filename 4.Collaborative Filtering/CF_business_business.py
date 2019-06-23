#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:19:54 2019

@author: Arnold Yu
@discription: This is a collabortive filtering recommendation system. Ranking based item based collabortive filtering.
"""

import pandas as pd
import numpy as np
from sortedcontainers import SortedList
import copy
import itertools

if __name__ == "__main__":
    
    
    matrix = pd.read_csv('user_business_preprocess.csv',  index_col = 0,  na_filter=False)
    matrix = matrix.T
    # number of neighbors we want to consider for time efficiency when have large dataset
    # 
    K = 20
    # number of common business users must have in common in order to consider
    limit = 5 
    
    matrix.shape

    #  business - user - rating
    dict_ = {}
    for row in matrix.iterrows():
        dict_[row[0]] = {}
        for col in matrix:
            if row[1][col] != '':
                dict_[row[0]][col] = row[1][col]
    
    # To aviod cold start
    # There are 924 users with 3 and higher reviews
    # There are 2601 users with 2 and higher reviews
    # There are 16008 users with 1 and higher reviews
    # This number had be test with 1, 2, 3
    numberOfReviews = 10
    dict_1 = {}
    for k,v in dict_.items():
        if len(v) >= numberOfReviews: 
            dict_1[k] = copy.deepcopy(v)
            
    
            
        
    

    def splitDict(d):
        n = len(d) // 5          # length of smaller half
        i = iter(d.items())      # alternatively, i = d.iteritems() works in Python 2
    
        d1 = dict(itertools.islice(i, n))   # grab first n items
        d2 = dict(i)                        # grab the rest
    
        return d1, d2
      
            
    
    def calculateWeight(dict_1):
        dict_2 = {}
        for k, v in dict_1.items():
            items_i = []
            items_set_i = set()
            counter_i = 0
            number_i = 0.0
            for i, j in v.items():
                items_i.append(i)
                items_set_i.add(i)
                number_i += float(j)
                counter_i += 1
            avg_i = float(number_i)/float(counter_i)
            dev_i = {}
            dev_i_values = []
            for i1, j1 in v.items():
                dev_i[i1] = float(j1) - avg_i
                dev_i_values.append(float(j1) - avg_i)
            dev_i_sum = 0.0
            for i in range(0,len(dev_i_values)):
                dev_i_sum += dev_i_values[i] ** 2
            sigma_i = np.sqrt(dev_i_sum)
            
            dict_2[k] = {'Average' : avg_i, 'Deviation': dev_i}
            sl = SortedList()
            for x,y in dict_1.items():
                if x != k:
                    items_j = []
                    items_set_j = set()
                    counter_j = 0
                    number_j = 0.0
                    for a, b in y.items():
                        items_j.append(a)
                        items_set_j.add(a)
                        number_j += float(b)
                        counter_j += 1
                    common_items = (items_set_i & items_set_j)                   
                    if len(common_items) >= limit:
                        #print('True')
                        avg_j = float(number_j)/float(counter_j)
                        dev_j = { item : float(rating) - avg_j for item, rating in y.items()}
                        dev_j_values = np.array(list(dev_j.values()))
                        sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))
                        
                        numerator = sum(dev_i[item]*dev_j[item] for item in common_items)
                        if sigma_i != 0.0 and sigma_j != 0.0:
                            w_ij = numerator / (sigma_i * sigma_j)                        
                            sl.add((-w_ij, x))
                        if len(sl) > K:
                          del sl[-1]
            dict_2[k]['Neighboor'] = sl
        return dict_2
    
    
    def predict(user, business, dict_2):
    # calculate the weighted sum of deviations
        numerator = 0.0
        denominator = 0.0
        #list_ = []
        for neg_w, j in dict_2[user]['Neighboor']:
            # remember, the weight is stored as its negative
            # so the negative of the negative weight is the positive weight           
            try:
                numerator += -neg_w * dict_2[j]['Deviation'][business]
                denominator += abs(neg_w)
            except KeyError:
                # neighbor may not have rated the same movie
                # don't want to do dictionary lookup twice
                # so just throw exception
                pass
        
        if denominator == 0.0:
            prediction = dict_2[user]['Average']
        else:
            # I didn't scale on 0 to 5 in order to separate ranking from user-business with same stars
            prediction = numerator / denominator + dict_2[user]['Average']
            #list_.append((numerator, denominator, dict_2[user]['Average']))
            #prediction = min(5, prediction)
            #prediction = max(0.5, prediction) # min rating is 0.5
            
        return prediction
    
    # root mean square error
    # since we don't scale on 0 to 5 
    # we need to adjust on that
    def rmse(dict1, dict3):
        # dict3 is a subset of dict1 for number of users
        dict_3 = {}
        for k, v in dict3.items():
            array = []
            if k in dict1:
                for i, j in dict1[k].items():
                    if i in dict3[k]:
                        if isinstance(dict3[k][i], float) and dict3[k][i] > 5.0 and isinstance(j, str):
                            array.append( (float(j) - 5.0) **2)
                        elif isinstance(dict3[k][i], float) and dict3[k][i]<= 5.0 and isinstance(j, str):
                            array.append( (float(j) - dict3[k][i] ) **2)
            if len(array) != 0:
                dict_3[k] = np.sqrt(np.mean(array))
            else:
                dict_3[k] = 0.0
        return dict_3
    
 
    dict_test, dict_train = splitDict(dict_1)
    dict_2_train = calculateWeight(dict_train) #0.8
    dict_2_test = calculateWeight(dict_test) # 0.2
    
    
    dict_3_train = {}
    dict_3_test = {}
    matrix.columns[1]
    
    for k, j in dict_2_train.items():
        dict_3_train[k] = {}
        for i in range(0, len(matrix.columns)):
            dict_3_train[k][matrix.columns[i]] = predict(k, matrix.columns[i], dict_2_train)
    for k, j in dict_2_test.items():
        dict_3_test[k] = {}
        for i in range(0, len(matrix.columns)):
            dict_3_test[k][matrix.columns[i]] = predict(k, matrix.columns[i], dict_2_test)

    dict_4_train = rmse(dict_1,dict_3_train)
    dict_4_test = rmse(dict_1, dict_3_test)
    array_train = []
    array_test = []
    for k,v in dict_4_train.items():
        array_train.append(v)
        
    for k, v in dict_4_test.items():
        array_test.append(v)
    rmse_average_train = np.mean(array_train)
    rmse_average_test = np.mean(array_test)
    
    # K = 20, limit = 5
    # 10 or more reviews
    # test rmse 1.362073964408062
    # train rmse 1.3307097267484014
    print("RMSE of Training Set : %f"  %rmse_average_train)
    print("RMSE of Test Set : %f" %rmse_average_test)
    
                
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
    
    
    def findTop10_and_Print(dict_):
        
        sorted_x = sorted(dict_.items(), key=lambda kv: kv[1], reverse = True)
        if len(sorted_x) >= 10:
            a11 = sorted_x[:10]
        else:
            a11 = sorted_x
        # store the top 10 buisness_id
        list_ = {}
        for i in range(0,len(a11)):
            list_[a11[i][0]] = i + 1
            
        for row in business.iterrows():
            if row[1]['business_id'] in list_:
                list_[row[1]['business_id']] = (list_[row[1]['business_id']] ,row[1]['name'] + row[1]['address'] + row[1]['city']+ row[1]['state'] + row[1]['postal_code'])              
       
        
        dict4 = {}
        for k,v in list_.items():
            dict4[v[0]] = v[1]
        for k,v in dict4.items():
            print(k,v)
         
        
    sum_ap = 0
    sum_ar = 0
    # turn business -> user dictionary to user-business dictionary
    dict_user = {}
    for k,v in dict_3_train.items():
        for k1,v1 in v.items():
            dict_user[k1] = {}
    for k,v in dict_3_train.items():
        for k1,v1 in v.items():
            dict_user[k1][k] = v1
    for k,v in dict_3_test.items():
        for k1,v1 in v.items():
            dict_user[k1][k] = v1
            
    for k, v in dict_user.items():
        result = calculateReleventR(v,10)
        ap = averagePercision(result)
        ar = averageRecall(result)
        sum_ap += ap
        sum_ar += ar

    # sum_ap = 15907.340714600641
    # sum_ar = 8817.461507937518
    meanap = sum_ap / float(matrix.shape[1]) #0.9937119386931935
    meanar = sum_ar / float(matrix.shape[1]) #0.5508159362779559
    print("MAP@10: %f" %meanap)
    print("MAR@10: %f" %meanar)
    
    # Predict user '-318sKiQDgbjLzF4FCU1XA'
    dict_5 = {}
    for k,v in dict_3_train.items():
        for k1, v1 in v.items():
            if k1 == "'-318sKiQDgbjLzF4FCU1XA'":
                dict_5[k] = v1
    for k,v in dict_3_test.items():
        for k1, v1 in v.items():
            if k1 == "'-318sKiQDgbjLzF4FCU1XA'":
                dict_5[k] = v1
    # print out top 10            
    findTop10_and_Print(dict_5)