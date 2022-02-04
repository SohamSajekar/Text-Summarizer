#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Import libraries
import re
import nltk
import spacy
import string
import pickle
import numpy as np
import pandas as pd
en = spacy.load('en_core_web_sm')
sw_spacy = en.Defaults.stop_words
from tqdm.notebook import tqdm_notebook
# Gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import re
from gensim.models import TfidfModel, LsiModel
from gensim.models.ldamodel import LdaModel
from gensim import matutils
from sklearn.cluster import KMeans
from collections import defaultdict
from Cleaning import data_preprocess

# Load models
lsa_model = pickle.load(open(r"Data/lsa_model_6.pkl", "rb"))
dictionary = pickle.load(open(r"Data/dictionary.pkl", "rb"))
corpus = pickle.load(open(r"Data/corpus.pkl", "rb"))
topic_dict = pickle.load(open(r"Data/topic_dict_lsa.pkl", "rb"))
texts = pickle.load(open(r"Data/texts.pkl", "rb"))


# In[2]:


# # Load stored data
# data_train = pd.read_csv(r"Data\\cleaned_training_data.csv")
# original_articles, cleaned_articles, cleaned_articles_merged = pickle.load(open(r"Data\\cleaned_training_data.pkl", "rb"))


# In[3]:


# n = 5000
# data_train = data_train[:n]
# original_articles = original_articles[:n]
# cleaned_articles = cleaned_articles[:n]
# cleaned_articles_merged = cleaned_articles_merged[:n]


# In[4]:


# print("data_train:",len(data_train))
# print("original_articles:",len(original_articles))
# print("cleaned_articles:",len(cleaned_articles))
# print("cleaned_articles_merged:",len(cleaned_articles_merged))


# In[5]:


# texts = [[word for word in article.split(" ")] for article in cleaned_articles_merged[:-1]]


# In[6]:


# dictionary = corpora.Dictionary(texts)
# # print(dictionary.id2token) ## to see the actual dictionary generated
# corpus = [dictionary.doc2bow(text) for text in texts]


# In[7]:


# # fit LSA model
# lsa_model = LsiModel(corpus=corpus, id2word=dictionary, num_topics=6)


# In[8]:


# # Create Topic-Word dictionary
# topic_dict = {}
# for topic in range(0, lsa_model.num_topics):
#     temp = {}
#     for token, score in lsa_model.show_topic(topic, topn=len(dictionary)):
#         if token.isalpha():
#             topic_dict[str(topic)+"_"+token] = score


# In[9]:


# # Store model
# pickle.dump(lsa_model, open(r"Data\\lsa_model_6.pkl", "wb"))
# pickle.dump(topic_dict, open(r"Data\\topic_dict_lsa.pkl", "wb"))


# In[8]:





# #### Sentence ranking

# In[7]:


def create_new_weights_lsa(topic_dict, assigned_topic):
    weights_dict =  {}
    for topic_word, score in topic_dict.items():
        topic, word = topic_word.split("_")
        if topic == assigned_topic:
            weights_dict[word] = score
    return weights_dict


# In[11]:


def score_sentences_lsa(weights_dict, original_articles, cleaned_articles):
    sen_score = {}
    for idx, sen in enumerate(cleaned_articles):
        if len(sen) > 3:
            score = 0
            for word in sen.split(" "):
                if word in weights_dict:
                    score += weights_dict[word]
            try:
                sen_score[original_articles[idx]] = score
            except:
                pass
    sen_score = sorted(sen_score, key=sen_score.get, reverse=True)
    return sen_score


# In[9]:


def lsa_transform(article, topn=5):
    # Cleaning
    original_articles, cleaned_articles, cleaned_articles_merged = data_preprocess([article])
#     cleaned_articles_merged = cleaned_articles_merged[-1]
    cleaned_articles = cleaned_articles[-1]
    original_articles = original_articles[-1]
    
    # Create a new corpus, made of previously unseen documents.
    texts_new = [[word for word in article.split(" ")] for article in cleaned_articles_merged]
    other_corpus = [dictionary.doc2bow(text) for text in texts_new]
    assigned_topic = max(lsa_model[other_corpus][0], key = lambda i : i[1])[0] 
    weights_dict = create_new_weights_lsa(topic_dict, assigned_topic)
    sen_score = score_sentences_lsa(weights_dict, original_articles, cleaned_articles)
    return sen_score[:topn]


# In[10]:


# summary = lsa_transform("article", topn=5)
# summary


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[33]:


# # Create a new corpus, made of previously unseen documents.
# texts_new = [[word for word in article.split(" ")] for article in [cleaned_articles_merged[-1]]]
# other_corpus = [dictionary.doc2bow(text) for text in texts_new]
# # unseen_doc = other_corpus[0]
# assigned_topic = max(lsa_model[other_corpus][0], key = lambda i : i[1])[0]    


# In[35]:


# sen_score = {}
# weights_dict = topic_dict["Topic_"+str(assigned_topic)]
# for idx, sen in enumerate(cleaned_articles[-1]):
#     if len(sen) > 3:
#         score = 0
#         for word in sen.split(" "):
#             if word in weights_dict:
#                 score += weights_dict[word]
#         sen_score[original_articles[-1][idx]] = score
# sen_score = sorted(sen_score, key=sen_score.get, reverse=True)


# In[36]:


# sen_score[:5]


# In[ ]:




