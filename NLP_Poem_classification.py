#!/usr/bin/env python
# coding: utf-8

# In[69]:


import pandas as pd                                #importing pandas library
import numpy as numpy                              
import random                                     # importing random for shuffling of data
import nltk
from nltk.corpus import stopwords                 # importing stopwords library for pre-processing  
from nltk.classify.scikitlearn import SklearnClassifier   
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC            



# In[70]:


df = pd.read_csv("all.csv")                        # reading data
# In[71]:

stop_words = set(stopwords.words("english"))       # Pre-processing - storing stopwords in list stop_words
stop_words.extend([',','.',':','-','!'])           # Adding punctuations to stop words list

poems = []                                         
for i in range(0,df.shape[0]):                     
        content = word_tokenize(df.at[i,'content'])    
        c = []                                     
        for w in content:
            if w not in stop_words:
                c.append(w.lower())
        poems.append((c[:110],df.at[i,'Type']))     # Normalization
                                                    # Mean of number of words is around 110, so only taking 110 words 
        


# In[73]:

filtered = []

for content in df['content']:
    for w in word_tokenize(content.lower()) :
        if w not in stop_words:
            filtered.append(w.lower())

filtered = nltk.FreqDist(filtered)                   # Calculating frequency of the words
word_features = list(filtered.keys())[:2300]         # Taking 2300 important words


# In[75]:


def find_features(document):                         # Finding feature dataset
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    
    return features                        

feature_sets = [(find_features(rev), category) for (rev, category) in poems]
random.shuffle(feature_sets)                         # Shuffling data for better training 
training_set = feature_sets[:df.shape[0]*0.85]		 # Taking 85% of the data for training
test_set = feature_sets[df.shape[0]*0.85:]           # Taking 15% of the data for testing

classifier = nltk.NaiveBayesClassifier.train(training_set)  # training by Naive bayes Classifier
accuracy = nltk.classify.accuracy(classifier,test_set)      # testing by Naive Bayes Classifier
print("Naive Bayes Classifier",accuracy)


MNB_classifier = SklearnClassifier(MultinomialNB())			# Multinomial Naive Bayes 
MNB_classifier.train(training_set)							
BNB_classifier = SklearnClassifier(BernoulliNB())			# Bernouli Naive Bayes
BNB_classifier.train(training_set)

acc = nltk.classify.accuracy(MNB_classifier , test_set)
print("MNB Classifer",acc)
acc = nltk.classify.accuracy(BNB_classifier, test_set)
print("BNB Classifer",acc)


Logistic_classifier = SklearnClassifier(LogisticRegression())   # Logistic Regression
Logistic_classifier.train(training_set)

acc = nltk.classify.accuracy(Logistic_classifier , test_set)
print("Logistic Classifer",acc)

SGD_classifier = SklearnClassifier(SGDClassifier())				# SGD Classifier
SGD_classifier.train(training_set)

acc = nltk.classify.accuracy(SGD_classifier , test_set)
print("SGD Classifer",acc)

SVC_classifier = SklearnClassifier(SVC())						# SVC Classifier
SVC_classifier.train(training_set)

acc = nltk.classify.accuracy(SVC_classifier , test_set)
print("SVC Classifer",acc)

LinearSVC_classifier = SklearnClassifier(LinearSVC())           # Linear SVC Classifier
LinearSVC_classifier.train(training_set)

acc = nltk.classify.accuracy(LinearSVC_classifier , test_set)
print("LinearSVC Classifer",acc)


# In[ ]:





# In[ ]:




