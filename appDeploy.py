# -*- coding: utf-8 -*-
"""
Created on Sun May  3 17:22:46 2020

@author: Medha
This code is mainly to test the app.py code without html/deploy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import nltk
import pickle 
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') 
from matplotlib import style;
style.use('ggplot')
stopwords = stopwords.words('english')


int_features = ' took pill yesterday noticed almost immediately appetite hummus wrap lunch couldnt even half usually whole thing dinner other half then took pill write almost hour later that pill wish never taken wont take again last chill tingled long thought signaling attack that thought caused could well twice bathroom hold wall there were severe pain this morning hard time dragging myself most med seem react fully cannot take belviq' 

int_features = int_features.lower()
int_features = int_features.replace('[^\w\s]', '')
int_features= int_features.replace('\d+', '')
int_features = int_features.replace(r'\b\w{1,3}\b', '')
int_features = re.sub(r'\s+[a-zA-Z]\s+', ' ', int_features)
int_features = re.sub(r'\W', ' ', int_features)
pattern = '[0-9]'
int_features =  re.sub(pattern, '', int_features) 

comment_words = ' '
tokens = []
tokens = int_features.split()
for val in tokens:
    if len(val) > 3:
        comment_words = comment_words + val + ' '
#            
int_features = ' '
int_features = comment_words
 
data = [int_features]
 
cvec = CountVectorizer(ngram_range=(1,2))
cvec.fit(data)
  
# Creating the bag-of-words representation
cvec_counts = cvec.transform(data)

# Instantiating the TfidfTransformer
transformer = TfidfTransformer()

# Fitting and transforming n-grams
transformed_weights = transformer.fit_transform(cvec_counts)
transformed_weights

# Getting a list of all n-grams
transformed_weights = transformed_weights.toarray()
vocab = cvec.get_feature_names()

# Putting weighted n-grams into a DataFrame and computing some summary statistics
X_test = pd.DataFrame(transformed_weights, columns=vocab)


#################################################
 
# One hot encoding to match the columns in test and train

one_hot_encoded_train =pd.read_csv('C:\\Users\\Medha\\Desktop\\Medha\\ExcelR\\Live Projects\\FinalApp\\encoded_train.csv')
one_hot_encoded_test = pd.get_dummies(X_test)


one_hot_encoded_train.shape
one_hot_encoded_test.shape

final_train, final_test = one_hot_encoded_train.align(one_hot_encoded_test,
                                                                    join='left', 
                                                                    axis=1)

 
X_Test = final_test
X_Test.shape

X_Test = X_Test.fillna(0)



model = pickle.load(open('model.pkl','rb'))
predicted= model.predict(X_Test)
print(predicted)
if  predicted == 'No':
   print('review does notcontain side effects')
else:
   print('review contains side effects')



 
 

 