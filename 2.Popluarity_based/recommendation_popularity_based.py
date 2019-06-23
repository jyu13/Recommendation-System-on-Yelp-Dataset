# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 14:47:56 2019

@author: Arnold
@discription: This is a simple recommendation system. Ranking based on popluarity
"""

# Importing the libraries
import numpy as np
import pandas as pd


class recommendation_Popluarity_Based:
    def __init__(self, category, city):
        self.category = category
        self.city = city
        self.business = pd.read_csv('../1.Preprocess/business_preprocess.csv')
        self.review = pd.read_csv('../1.Preprocess/review_preprocess.csv')
    
    def findBusiness(self):
        
        df = pd.DataFrame()        
        for row in self.business.iterrows():
            if self.category in row[1]['categories'].lower() and self.city in row[1]['city'].lower():
                df =pd.concat([df, pd.DataFrame(row[1]).T])
                
        dict_ = {}
        for row in df.iterrows():
            dict_[row[1]['business_id']] = {'Negative': 0.0,'Positive': 0.0}
            
        return dict_

    
    
    def ratingBased(self, dict_):
        
        for row in self.review.iterrows():
            if row[1]['business_id'] in dict_:
                id_ = row[1]['business_id']
                if row[1]['stars'] == 0.0 or row[1]['stars'] == 0:
                    dict_[id_]['Negative'] += 1.0
                    dict_[id_]['Positive'] += 0.0
                if row[1]['stars'] == 0.5:
                    dict_[id_]['Negative'] += 0.9
                    dict_[id_]['Positive'] += 0.1
                if row[1]['stars'] == 1.0 or row[1]['stars'] == 1:
                    dict_[id_]['Negative'] += 0.8
                    dict_[id_]['Positive'] += 0.2
                if row[1]['stars'] == 1.5:
                    dict_[id_]['Negative'] += 0.7
                    dict_[id_]['Positive'] += 0.3
                if row[1]['stars'] == 2.0 or row[1]['stars'] == 2:
                    dict_[id_]['Negative'] += 0.6
                    dict_[id_]['Positive'] += 0.4
                if row[1]['stars'] == 2.5:
                    dict_[id_]['Negative'] += 0.5
                    dict_[id_]['Positive'] += 0.5
                if row[1]['stars'] == 3.0 or row[1]['stars'] == 3:
                    dict_[id_]['Negative'] += 0.4
                    dict_[id_]['Positive'] += 0.6
                if row[1]['stars'] == 3.5:
                    dict_[id_]['Negative'] += 0.3
                    dict_[id_]['Positive'] += 0.7
                if row[1]['stars'] == 4.0 or row[1]['stars'] == 4:
                    dict_[id_]['Negative'] += 0.2
                    dict_[id_]['Positive'] += 0.8
                if row[1]['stars'] == 4.5:
                    dict_[id_]['Negative'] += 0.1
                    dict_[id_]['Positive'] += 0.9
                if row[1]['stars'] == 5.0 or row[1]['stars'] == 5:
                    dict_[id_]['Negative'] += 0.0
                    dict_[id_]['Positive'] += 1.0
    
        return dict_
    
    
    
    
    def findTop10_and_Print(self,dict_):
        
        dict_1 = {}        
        for k,v in dict_.items():
            dict_1[k] = v['Positive'] - v['Negative']
            
        sorted_x = sorted(dict_1.items(), key=lambda kv: kv[1], reverse = True)
        if len(sorted_x) >= 10:
            a11 = sorted_x[:10]
        else:
            a11 = sorted_x
        
        # store the top 10 buisness_id
        list_ = {}
        for i in range(0,len(a11)):
            list_[a11[i][0]] = i + 1
            
        for row in self.business.iterrows():
            if row[1]['business_id'] in list_:
                list_[row[1]['business_id']] = (list_[row[1]['business_id']] ,row[1]['name'] + row[1]['address'] + row[1]['city']+ row[1]['state'] + row[1]['postal_code'])              
        
        dict4 = {}
        for k,v in list_.items():
            dict4[v[0]] = v[1]
        for k,v in dict4.items():
            print(k,v)
    
    def calculateReleventR(self, dict_, N):

        list_ = {}
        dict_1 = {}  
        relevent = 0
        for k,v in dict_.items():
            dict_1[k] = v['Positive'] - v['Negative']
            if dict_1[k] >= 100:
                relevent += 1
                        
        sorted_x = sorted(dict_1.items(), key=lambda kv: kv[1], reverse = True)

        if len(sorted_x) >= N:
            a11 = sorted_x[:N]
        else:
            a11 = sorted_x
                    
        for i in range(0,len(a11)):
            list_[a11[i][0]] = i
                
        result = [0] * N
        for row in self.business.iterrows():
            if row[1]['business_id'] in list_:
                if row[1]['stars'] >= 3.5:
                    result[list_[row[1]['business_id']]] = 1
        return result, relevent
    
    def averagePercision(self, list_):
        m = sum(list_)
        k = len(list_)
        ap = 0.0
        for i in range(0,k):
            ap += float(sum(list_[0:i+1]))/float(len(list_[0:i+1])) *list_[i]
        return ap/float(m)
        
    def averageRecall(self, list_):
        
        m = sum(list_)
        k = len(list_)
        ar = 0.0
        for i in range(0,k):
            ar += float(sum(list_[0:i+1]))/float(m) *list_[i]
            
        return ar/float(m)
    
    
if __name__ == "__main__":
    
    a = recommendation_Popluarity_Based('Chinese'.lower(), 'Phoenix'.lower())    
    dict_ = a.findBusiness()
    dict_1 = a.ratingBased(dict_)   
    result, _ = a.calculateReleventR(dict_1, 10)
    ap = a.averagePercision(result) # 1.0
    ar = a.averageRecall(result) # 0.55
    #MAP@N is same as AP@N since they are the same for all users
    print("MAP@10: %f" %ap)
    print("MAR@10: %f" %ar)
    a.findTop10_and_Print(dict_1)
    
    """busy = pd.read_csv('../1.Preprocess/business_preprocess.csv')
    df = pd.DataFrame()        
    for row in busy.iterrows():
        if 'Chinese'.lower() in row[1]['categories'].lower() and 'Phoenix'.lower() in row[1]['city'].lower():
            df =pd.concat([df, pd.DataFrame(row[1]).T])"""
    