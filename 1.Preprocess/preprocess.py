# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 18:22:07 2019

@author: Arnold Yu
@discription: This script preprocess the datasets, drop the columns and rows containing nan.
"""

import pandas as pd


# Preprocess business.csv
business = pd.read_csv('../0.Data/business.csv')
business.shape
# (192609, 60)
business_column = [ 'business_id','name','address','city','state','postal_code', 'categories','review_count','stars','latitude','longitude']
business = pd.DataFrame(business, columns = business_column )
business = business.dropna(axis= 0)

business = business.iloc[:,:].values

for i in range(0, len(business)):
    for j in range(0, len(business[i])):
        if isinstance(business[i][j], str):
            business[i][j] = business[i][j][1:]

business = pd.DataFrame(business, columns = business_column)
business.shape
# (192127, 11)
business.to_csv('business_preprocess.csv', sep = ",",index = False)



# Preprocess review.csv
review = pd.read_csv('../0.Data/review.csv')
review.shape
# (6685900, 9)
# review.iloc[1,:]
review_column = ['business_id', 'stars', 'text', 'user_id' ]
review = pd.DataFrame(review, columns = review_column)
review = review.dropna(axis= 0)

review = review.iloc[:,:].values

for i in range(0, len(review)):
    for j in range(0, len(review[i])):
        if isinstance(review[i][j], str):
            review[i][j] = review[i][j][1:]

review = pd.DataFrame(review, columns = review_column)
review.shape
# (6685900, 4)
review.to_csv('review_preprocess.csv', sep = ",",index = False)

review['business_id'][1]
review['stars'][1]
review['text'][1]

# Preprocess user.csv
user = pd.read_csv('../0.Data/user.csv')
# user.shape
# (1637138, 22)
# user.iloc[1, :]
user_column = ['review_count',  'user_id', 'average_stars', 'friends' ]
user = pd.DataFrame(user, columns = user_column)
user = user.dropna(axis= 0)

user = user.iloc[:,:].values

for i in range(0, len(user)):
    for j in range(0, len(user[i])):
        if isinstance(user[i][j], str):
            user[i][j] = user[i][j][1:]

user = pd.DataFrame(user, columns = user_column)
# user.shape
# (1637138, 4)
user.to_csv('user_preprocess.csv', sep = ",",index = False)