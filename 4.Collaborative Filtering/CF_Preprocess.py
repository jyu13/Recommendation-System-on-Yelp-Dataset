#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 14:04:25 2019

@author: Arnold Yu
@discription: This is a collabortive filtering recommendation system. Build user_business matrix
"""
import math
import collections
import numpy as np
import pandas as pd
import sys


if __name__ == "__main__":
    
    business = pd.read_csv('../1.Preprocess/business_preprocess.csv')
    review = pd.read_csv('../1.Preprocess/review_preprocess.csv')
    # user = pd.read_csv('../1.Preprocess/user_preprocess.csv')
    category = 'Chinese'.lower()
    city = 'Phoenix'.lower()
    
    
    
       
    # build business and user matrix

    df = pd.DataFrame()        
    for row in business.iterrows():
        if category in row[1]['categories'].lower() and city in row[1]['city'].lower():
            df =pd.concat([df, pd.DataFrame(row[1]).T])
         
    df.to_csv('business_preprocess_filtered.csv',sep=',')
    # df['business_id'][78]
        
    dict_ = {}
    for row in df.iterrows():
        dict1 = {}
        dict_[row[1]['business_id']] = dict1
            
        
    for row in review.iterrows():
        if row[1]['business_id'] in dict_:
            dict_[row[1]['business_id']][row[1]['user_id']] = row[1]['stars']
    
    
    column = []
    row = set()
    for k, v in dict_.items():
        column.append(k)
        for i,j in v.items():
            row.add(i)
    #20866
    row = list(row)
    #16008
    df1 = pd.DataFrame(index = row, columns = column)
    # df1[column][row]
    for k, v in dict_.items():
        for i, j in v.items():
            df1[k][i] = j
            
    df1.to_csv('user_business_preprocess.csv', sep = ",")
    