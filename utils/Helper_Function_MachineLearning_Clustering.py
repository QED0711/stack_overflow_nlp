#!/usr/bin/env python
# coding: utf-8

# In[6]:


# import necessary libraries
from collections import Counter
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import re
from gensim import corpora, models
import gensim
import warnings
warnings.filterwarnings('ignore')


# In[5]:


def confusion_matrix_df(modelname, modelestimator, pred_value):
    '''
    input: 
    modelname: classifier contains true value and class labels
    return:
    confusion matrix dataframe
    '''
    cm = confusion_matrix(modelname._y_test,pred_value, labels= modelestimator.classes_)
    cm_df = pd.DataFrame(cm, columns=modelestimator.classes_+' Pred',
             index=modelestimator.classes_+' True')
    return cm_df


# In[3]:


def majority_vote(lg_pred,rf_pred,nb_pred):
    '''
    inputs:
    lg_pred:
    rf_pred:
    nb_pred:
    returns:
    combined prediction result 
    '''
    combined = []
    predicted = zip(lg_pred,rf_pred,nb_pred)
    for p in predicted:
        votes = Counter(p)
        most = max(votes.values())
        if most >1:
            combined.append(list(votes.keys())[list(votes.values()).index(most)])
        else:
            combined.append(np.random.choice(list(votes.keys()),1)[0])
    return combined


# In[4]:


def coef_features(modelname, lg = True):
    '''
    inputs:
    modelname: models  
    lg: if it is linear regression if not it is nb
    returns:
    a dictionary of coefs and featurename 
    '''
    labels = modelname.classes_  # label 
    if lg:
        coefs = modelname[1].coef_
    else:
        coefs = modelname[1].feature_log_prob_
    featurenames = modelname[0].get_feature_names()
    coef_dict={}
    for i, l in enumerate(labels):
        coef_dict[l]=[]
        for c, f in zip(coefs[i],featurenames):
            if c:
                coef_dict[l].append((f,c))
    return coef_dict   

def top_10_feature(coef_dict, lg = True):
    '''
    input:
    coef_dict: a dictionary of coefs and featurename from coef_features function
    lg: if true it is logistic regression otherwise nb 
    returns:
    a data frame of top 10 most important features for each classes 
    '''
    top_10 = {}
    for l in coef_dict.keys():
        top_10[l] = sorted(coef_dict[l], key = lambda x: x[1], reverse = True)[:10]
        top_10[l] = [x[0] for x in top_10[l]]
    df = pd.DataFrame(top_10)
    if lg:
        df.columns = df.columns + '_lg'
    else:
        df.columns = df.columns + '_nb'
    return df


# In[7]:


def topic_top_word(model):
    '''
    input:
    model: lda_model (bow or tfidf)
    return:
    a dataframe with top words for each topic 
    '''
    topics= model.print_topics(num_topics=5,num_words=5) 
    topics_dict = {}
    for topic in topics:
        topics_dict[topic[0]] = re.findall('[a-z]+',topic[1])
    df = pd.DataFrame(topics_dict)
    df.columns = ['topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4']
    return df


# In[8]:


def text_topic(df, model,topic):
    '''
    input:
    df: raw text, lda model, and topic num
    returns:
    random text for that topic 
    '''
    texts = df[df[model]==topic].text
    inds = df[df[model]==topic].text.index
    ind = np.random.choice(inds)
    return texts[ind]

