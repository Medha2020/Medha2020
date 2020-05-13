# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 18:58:00 2020

@author: Medha
"""
from flask import Flask,request,render_template
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


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb')) 

@app.route('/')
def home():
    return render_template('deployMed.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
        For rendering results on HTML GUI
    '''
    int_features = request.form['review']
    reviewText = int_features
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
    
    one_hot_encoded_train =pd.read_csv('C:\\Users\\Medha\\Desktop\\Medha\\ExcelR\\Live Projects\\app\\encoded_train.csv')
    one_hot_encoded_test = pd.get_dummies(X_test)
    
    
    one_hot_encoded_train.shape
    one_hot_encoded_test.shape
    
    final_train, final_test = one_hot_encoded_train.align(one_hot_encoded_test,
                                                                        join='left', 
                                                                        axis=1)
    
     
    X_Test = final_test
    X_Test.shape
    
    X_Test = X_Test.fillna(0)
    
    
    
#    model = pickle.load(open('model.pkl','rb'))
    predicted= model.predict(X_Test)
    
#    int_features = request.get('review')
    if predicted == 'No':
       output = 1    
    elif predicted == 'Yes':
       output = 2

#       

#    return  render_template('deployMed.html',prediction_text = 'Drug Review Rating contains $ {}'.format(output))
    return render_template('result.html',prediction = output,textOutput = reviewText)
 
if __name__ == '__main__':
	app.run(debug=True)