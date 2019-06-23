# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 23:39:52 2019

@author: Arnold Yu
@discription: This is a simple recommendation system. Ranking based on content of user reviews.
"""

# Importing the libraries
import math
import collections
import numpy as np
import pandas as pd
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import sys

class recommendation_Content_Based:
    def __init__(self, category, city, content):
        self.category = category.lower()
        self.city = city.lower()
        self.business = pd.read_csv('../1.Preprocess/business_preprocess.csv')
        self.review = pd.read_csv('../1.Preprocess/review_preprocess.csv')
        # only keep letters
        text = re.sub('[^a-zA-Z]', ' ', content)
        text = text.lower()
        text = text.split()
        ps = PorterStemmer()
        # keep important words and its stem only
        text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
        self.text = text
    
    def findBusiness(self):
        
        df = pd.DataFrame()        
        for row in self.business.iterrows():
            if self.category in row[1]['categories'].lower() and self.city in row[1]['city'].lower():
                df =pd.concat([df, pd.DataFrame(row[1]).T])
                
        dict_ = {}
        for row in df.iterrows():
            dict1 = {}
            for i in range(0,len(self.text)):
                dict1[self.text[i]] = 0 
            dict_[row[1]['business_id']] = dict1
           
        return dict_
    
    
    def contentBased(self, dict_):
        
        # Cleaning the texts       
        for row in self.review.iterrows():
            if row[1]['business_id'] in dict_:
                # only keep letters
                text = re.sub('[^a-zA-Z]', ' ', row[1]['text'])
                text = text.lower()
                text = text.split()
                ps = PorterStemmer()
                # keep important words and its stem only
                text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
                for i in range(0,len(text)):
                    if text[i] in dict_[row[1]['business_id']]:
                        dict_[row[1]['business_id']][text[i]] += 1
            
        return dict_
    
    # Apply TFIDF to the keywords
    def TFIDF(self, dict_):
        df = pd.DataFrame()
        for k, v in dict_.items():
            array = []
            array.append(k)
            for i,j in v.items():
                array.append(j)
            df = pd.concat([df, pd.DataFrame(array).T], ignore_index=True)
        
        dict2 = {}
        for i in range(1,df.shape[1]):
            number = 0
            max_ = max(df[i])
            min_ = min(df[i])
            for j in range(0,df.shape[0]):
                if df[i][j] != 0:
                    number += 1
            dict2[i] = {'maxN':max_, 'minN':min_ , 'number': number}
            
        # normalize and times idf
        for i in range(1,df.shape[1]):
            for j in range(0, df.shape[0]):
                df[i][j] = (float(df[i][j]) - float(dict2[i]['minN']))/(float(dict2[i]['maxN']) - float(dict2[i]['minN'])) * math.log10(df.shape[0]/dict2[i]['number'])
                             
        # df[column][row]
        # key = buisness_id  value = store that buiness get
        dict3 = collections.defaultdict(int)   
        for j in range(0, df.shape[0]):
            number = 0
            for i in range(1, df.shape[1]):
                number += df[i][j] ** 2
            dict3[df[0][j]] = number ** 0.5
        return dict3
    
    
    def findTop10_and_Print(self, dict_):
        
        sorted_x = sorted(dict_.items(), key=lambda kv: kv[1], reverse = True)
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
        sorted_x = sorted(dict_.items(), key=lambda kv: kv[1], reverse = True)
        if len(sorted_x) >= N:
            a11 = sorted_x[:N]
        else:
            a11 = sorted_x
        # store the top 10 buisness_id
        list_ = {}
        for i in range(0,len(a11)):
            list_[a11[i][0]] = i
                
        result = [0] * N
        for row in self.business.iterrows():
            if row[1]['business_id'] in list_:
                if row[1]['stars'] >= 3.5:
                    result[list_[row[1]['business_id']]] = 1
        return result
    
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
    
    category = 'Chinese'
    city = 'Phoenix'
    content = "I like fresh food, quick order. I also like chinese chicken soup."
    
    
    a = recommendation_Content_Based(category, city, content)
    dict_ = a.findBusiness()
    if dict_ == None:
        print('Can\'t find any matching business.')
        sys.exit()
        
        
    dict_ = a.contentBased(dict_)
    dict2 = a.TFIDF(dict_)
    
    result = a.calculateReleventR(dict2, 10)
    ap = a.averagePercision(result) # 1.0
    ar = a.averageRecall(result) # 0.55
    #MAP@N is same as AP@N since they are the same for all users
    print("MAP@10: %f" %ap)
    print("MAR@10: %f" %ar)
    a.findTop10_and_Print(dict2)


    
    
    """df = pd.DataFrame()        
    for row in business.iterrows():
        if category in row[1]['categories'].lower() and city in row[1]['city'].lower():
            df =pd.concat([df, pd.DataFrame(row[1]).T])                
    dict_ = {}
    for row in df.iterrows():
        dict_[row[1]['business_id']] = {}
            
    """
    
    """dict1 = {}
    text = re.sub('[^a-zA-Z]', ' ', review['text'][1])
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    # keep important words and its stem only
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    for i in range(0,len(text)):
        if text[i] not in dict1:
            dict1[text[i]] = 1
        else:
            dict1[text[i]] += 1
    """
    
    
    
    """content = "I like fresh food, quick order. I also like chinese chicken soup."
    # only keep letters
    text = re.sub('[^a-zA-Z]', ' ', content)
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    # keep important words and its stem only
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    dict1 = {}
    for i in range(0,len(text)):
        dict1[text[i]] = 0
        
    for k in dict_.keys():
        dict_[k] = dict1
        
        
    # Computer term frequency using dictionary
    for row in review.iterrows():
        if row[1]['business_id'] in dict_:
            # only keep letters
            text = re.sub('[^a-zA-Z]', ' ', row[1]['text'])
            text = text.lower()
            text = text.split()
            ps = PorterStemmer()
            # keep important words and its stem only
            text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
            for i in range(0,len(text)):
                if text[i] in dict_[row[1]['business_id']]:
                    dict_[row[1]['business_id']][text[i]] +=1

        """
    # In order to use content-based recommender
    # We need to give a basic content for searching
    # We will compute similiarity( between the content entried and keywords in reviews from each buiness
    
    """content = "I like fresh food, quick order. I also like chinese chicken soup."
    # only keep letters
    text = re.sub('[^a-zA-Z]', ' ', content)
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    # keep important words and its stem only
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    row = ['business_id'] + text
    """
    
    # Compute TFM
    # matrix = pd.DataFrame(columns = row)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    