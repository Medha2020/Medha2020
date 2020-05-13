# -*- coding: utf-8 -*-
"""
Created on Sun May  3 17:22:46 2020

@author: Medha
This code has all the models that were tested for comparison.
This is also used to generate plots and test single review input
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import nltk
import pickle 
import re
from nltk.corpus import stopwords
import seaborn as sns
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import model_selection, preprocessing, metrics

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') 
from matplotlib import style;
style.use('ggplot')
stopwords = stopwords.words('english')

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()   
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
 

df1 =pd.read_csv("C:\\Users\\Medha\\Desktop\\Medha\\ExcelR\\Live Projects\\Medicine Side Effects\\DataSet\\test.csv")
 
df1["Category"] = ' '
df1.columns
df1.loc[df1['rating'] > 7, 'Category'] = 'No' 
df1.loc[df1['rating'] <= 7, 'Category'] = 'Yes'

# create a new dataframe with just review and rating for sentiment analysis
df2 = df1[['Id','condition','review','Category']].copy()
df2.shape




dfUniqueCond = pd.DataFrame(df2.condition.unique(),columns=['condition'])
dfUniqueCond = dfUniqueCond.dropna(subset=['condition'])

#dfUniqueCond = pd.DataFrame(dfCond.condition.unique(),columns=['condition'])
dfUniqueCond['cleanCondition'] = dfUniqueCond['condition'].apply(lambda x: " ".join(x.lower() for x in x.split()))
dfUniqueCond['cleanCondition'] = dfUniqueCond['cleanCondition'].str.replace('\d+', '')
dfCondList = []
for index, row in dfUniqueCond.iterrows():
#     print(row['cleanCondition']) 
    text = row["cleanCondition"].split()
    for i in range(len(text)): 
        dfCondList.append(text[i])


dfCondList.extend(['side','effect','year','medication','medicine'])
 


 # remove stopwords from review
df2['cleanReview'] = df2['review'].apply(lambda x: ' '.join([item for item in x.split() if item not in stopwords]))  
df2['cleanReview'] = df2['review'].apply(lambda x: ' '.join([item for item in x.split() if item not in dfCondList]))  
   
df2['cleanReview'] = df2['cleanReview'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df2['cleanReview'] = df2['cleanReview'].apply(lambda x: "".join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))

df2['cleanReview'] = df2['cleanReview'].str.replace('[^\w\s]', '')
df2['cleanReview'] = df2['cleanReview'].str.replace('\d+', '')
df2['cleanReview'] = df2['cleanReview'].str.replace(r'\b\w{1,3}\b', '')
                     
df2['cleanReview'] = df2['cleanReview'].apply(lambda x: " ".join(x.strip() for x in x.split()))
df2['cleanReview'] = df2['cleanReview'].apply(lemmatize_text)
df_all = df2

 
comment_words = ' '
filtered_list= []
for row in df2.itertuples(index = True, name ='Pandas'):

    review = getattr(row, "cleanReview") 
#    print(review)
#    filtered_list= []
#    print('newList\n')
    for val in review:
#        print(val)
        if len(val) > 2:
            comment_words = comment_words + val + ' '
#            print(comment_words)
    filtered_list.append(comment_words) 
    comment_words = ' '
    review = ' '
filtered_list = pd.DataFrame(filtered_list)
filtered_list.shape
df2.shape
df_all['filtered_list'] = filtered_list

 
df_all.columns 
y = df_all['Category']  

y.shape 
 
from imblearn.over_sampling import SMOTE
smote=SMOTE()

################## Plots##########################################
#condition_dn = df1.groupby(['condition'])['drugName'].nunique().sort_values(ascending=False)
#condition_dn[0:20].plot(kind="bar", figsize = (14,6), fontsize = 10,color="blue")
#plt.xlabel("", fontsize = 20)
#plt.ylabel("", fontsize = 20)
#plt.title("Top20 : The number of drugs per condition.", fontsize = 20)
#
#percent = (df1.isnull().sum()).sort_values(ascending=False)
#percent.plot(kind="bar", figsize = (14,6), fontsize = 10, color='blue')
#plt.xlabel("Columns", fontsize = 20)
#plt.ylabel("", fontsize = 20)
#plt.title("Total Missing Value ", fontsize = 20)

#import plotly.graph_objs as go
#from collections import defaultdict
#df_all_Yes = df_all[df_all["Category"]=='Yes']
#df_all_No = df_all[df_all["Category"]=='No']
#
### custom function for ngram generation ##
#def generate_ngrams(text, n_gram=1):
#    token = [token for token in text.lower().split(" ") if token != "" if token not in stopwords]
#    ngrams = zip(*[token[i:] for i in range(n_gram)])
#    return [" ".join(ngram) for ngram in ngrams]
#
### custom function for horizontal bar chart ##
#def horizontal_bar_chart(df, color):
#    trace = go.Bar(
#        y=df["word"].values[::-1],
#        x=df["wordcount"].values[::-1],
#        showlegend=False,
#        orientation = 'h',
#        marker=dict(
#            color=color,
#        ),
#    )
#    return trace
#
### Get the bar chart from rating  8 to 10 review ##
#freq_dict = defaultdict(int)
#for sent in df_all_No["filtered_list"]:
#    for word in generate_ngrams(sent):
#        freq_dict[word] += 1
#fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
#fd_sorted.columns = ["word", "wordcount"]
#trace0 = horizontal_bar_chart(fd_sorted.head(25), 'blue')
#
### Get the bar chart from rating  4 to 7 review ##
#freq_dict = defaultdict(int)
#for sent in df_all_Yes["filtered_list"]:
#    for word in generate_ngrams(sent):
#        freq_dict[word] += 1
#fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
#fd_sorted.columns = ["word", "wordcount"]
#trace1 = horizontal_bar_chart(fd_sorted.head(25), 'blue')
#
#import plotly
#
#
## Creating two subplots
#fig = plotly.tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,
#                          subplot_titles=["Frequent words of rating No", 
#                                          "Frequent words of rating Yes"])
#fig.append_trace(trace0, 1, 1)
#fig.append_trace(trace1, 1, 2)
#fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
#plotly.offline.plot(fig, filename='word-plots')
#############################################################################
df_all_Yes = df_all[df_all["Category"]=='Yes']
df_all_No = df_all[df_all["Category"]=='No']

yes_string = []
no_string = []
 

for t in df_all_Yes.filtered_list:
    yes_string.append(t)
yes_string = pd.Series(yes_string).str.cat(sep=' ')

for t in df_all_No.filtered_list:
    no_string.append(t)
no_string = pd.Series(no_string).str.cat(sep=' ')


from wordcloud import WordCloud
######### Yes Side Effects  Word Cloud  ##################
wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(yes_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show() 

######### No Side Effects Word Cloud  ##################
wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(no_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show() 



##############################################################33333

############## Tfidf Only  and count vectorizer ###################################

df_all.shape
df_all_dup = df_all
df_all_dup.shape
#df_all= df_all_dup
#df_all= df_all_dup.head(8000)
 
cvec = CountVectorizer(min_df=5, ngram_range=(1,2),max_features=15000)
cvec.fit(df_all['filtered_list'])

# Creating the bag-of-words representation
cvec_counts = cvec.transform(df_all['filtered_list'])

# Instantiating the TfidfTransformer
transformer = TfidfTransformer()

# Fitting and transforming n-grams
transformed_weights = transformer.fit_transform(cvec_counts)
transformed_weights

# Getting a list of all n-grams
transformed_weights = transformed_weights.toarray()
vocab = cvec.get_feature_names()

# Putting weighted n-grams into a DataFrame and computing some summary statistics
X = pd.DataFrame(transformed_weights, columns=vocab)

X.shape
y = df_all['Category']  
y.shape

###########################################3


int_features = 'I&#039;ve been on Aviane for 11 days now, starting the 1st pill in the pack the first day my period started (Sunday). I&#039;ve now had my period a full 11 days and it&#039;s starting to get really annoying. My sex drive has decreased a bit, I have vaginal itching, burning and redness, and I find myself becoming frequently dizzy. My breasts have become extra sensitive and seem much fuller than normal, my acne/ clearing of face seems the same as before taking the pill, but my mood swings have been the worst. One moment I want to cry, the next moment I&#039;m cheerful and happy towards everyone.' 
#int_features  = review

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
#    print(val)
    if len(val) > 3:
        comment_words = comment_words + val + ' '
#            print(comment_words)
int_features = ' '
int_features = comment_words
 
#int_features = int_features.join(x.strip() for x in int_features.split())
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
#one_hot_encoded_train = pd.get_dummies(X)
one_hot_encoded_train =pd.read_csv('C:\\Users\\Medha\\Desktop\\Medha\\ExcelR\\Live Projects\\app\\encoded_test.csv')

one_hot_encoded_test = pd.get_dummies(X_test)

del X_train
del X_test

one_hot_encoded_train.shape
one_hot_encoded_test.shape

final_train, final_test = one_hot_encoded_train.align(one_hot_encoded_test,
                                                                    join='left', 
                                                                    axis=1)

y_sm = df_all['rating']
X_sm =  final_train
X_Test = final_test
y_sm.shape
X_sm.shape
X_Test.shape



X_Test.apply(lambda x: x.count(), axis=1)
X_Test.isnull().count()
X_Test.head(5)
X_Test = X_Test.fillna(0)
X_Test.replace(np.nan,0)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

X_sm, y_sm = smote.fit_sample(X_train,y_train)

X_train1, X_test1 = train_test_split(
    X_TestData, test_size=0.1, random_state=1)
 

###############  Countvectorizer  and tfidf calculated teogether #######################
##########################Model based on count vectoriser and tfidf  ######################
#########################################################################################
#Code to generate output file
#########################################################################################
#Ceate test file output

#modelAll = open('C:\\Users\\Medha\\Desktop\\Medha\\ExcelR\\Live Projects\\FinalApp\\modelRF.pkl','rb') 
#model =  pickle.load(modelAll)
#predictd = model.predict(X)
#predictd = pd.DataFrame(predictd)
#output = []
#output = pd.DataFrame(output,columns = ['id','output'])
#output['id'] = df2['Id']
#output['output'] = predictd
#output.to_csv('C:\\Users\\Medha\\Desktop\\Medha\\ExcelR\\Live Projects\\Medicine Side Effects\\DataSet\\output.csv',index = False)
#del output
#clf = LinearSVC().fit(X_sm, y_sm)
#for index, row in output.iterrows(): 
##    print(row['id'],row['output'])
#    if int(row['rating'] > 7):
#       output['output'].values[index] = 'No'
#      
#    else:
#       output['output'].values[index] = 'Yes'
#output = output.drop(['rating'],axis=1)
#########################################################################################



clf = LinearSVC().fit(X_train, y_train)

pickle.dump(clf,open('modelRF.pkl','wb'))

predicted= clf.predict(X_test)
print("LinearSVC Test Accuracy:",metrics.accuracy_score(y_test, predicted))
predicted= clf.predict(X_train)
print("LinearSVC Test Data Accuracy:",metrics.accuracy_score(y_train, predicted))
predicted= clf.predict(X_sm)
print("LinearSVC Train Accuracy:",metrics.accuracy_score(y_sm, predicted))
#cm = confusion_matrix(predicted, y_sm)
#print("Train Confusion Matrix:  \n", cm)
print("                    Train Classification Report \n",classification_report(predicted, y_train))
    
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Test Accuracy:",metrics.accuracy_score(y_test, predicted))
predicted= clf.predict(X_train)
print("MultinomialNB Train Accuracy:",metrics.accuracy_score(y_train, predicted))
#cm = confusion_matrix(predicted, y_train)
#print("Train Confusion Matrix:  \n", cm)
print("                    Train Classification Report \n",classification_report(predicted, y_test))
    

clf = LogisticRegression().fit(X_train, y_train)
pickle.dump(clf,open('modelLog.pkl','wb'))
predicted= clf.predict(X_test)
print("LogisticRegression Test Accuracy:",metrics.accuracy_score(y_test, predicted))
predicted= clf.predict(X_train)
print("LogisticRegression Train Accuracy:",metrics.accuracy_score(y_train, predicted))
#cm = confusion_matrix(predicted, y_train)
#print("Train Confusion Matrix:  \n", cm)
print("                    Train Classification Report \n",classification_report(predicted, y_train))
 
clf = RandomForestClassifier().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("RandomForestClassifier Test Accuracy:",metrics.accuracy_score(y_test, predicted))
predicted= clf.predict(X_train)
    print("RandomForestClassifier Train Accuracy:",metrics.accuracy_score(y_train, predicted))
cm = confusion_matrix(predicted, y_train)
print("Train Confusion Matrix:  \n", cm)
print("                    Train Classification Report \n",classification_report(predicted, y_train))


 
 

 