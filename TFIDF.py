#!/usr/bin/env python
# coding: utf-8

# In[26]:


import math
import pickle
import nltk
import pandas as pd 
from operator import itemgetter
from nltk import sent_tokenize, word_tokenize, PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from Cleaning import data_preprocess

count_vect_model = pickle.load(open(r"Data/count_vect_model.pkl", "rb"))
tfidf_model = pickle.load(open(r"Data/tf_idf_model.pkl", "rb"))


# ### Training Model

# In[27]:


# data_train = pd.read_csv(r"Data/cleaned_training_data.csv")
# original_articles, cleaned_articles, cleaned_articles_merged = pickle.load(open(r"Data/cleaned_training_data.pkl", "rb"))


# In[28]:


# # Function definition to Train TF-IDF Model
# def tf_idf_train(cleaned_articles_merged):
#     count_vect = CountVectorizer()
#     tfidf = TfidfTransformer(norm="l2")
#     count_vect_model = count_vect.fit(cleaned_articles_merged)
#     freq_term_matrix = count_vect_model.transform(cleaned_articles_merged)
#     tf_idf_model= tfidf.fit(freq_term_matrix)
#     return count_vect_model, tf_idf_model


# In[29]:


# # Training,Saving the Models
# count_vect_model, tf_idf_model =tf_idf_train(cleaned_articles_merged) # Training
# pickle.dump(count_vect_model, open(r"C:\Users\PRAVEEN\Desktop\MSAI\1. University Of Georgia, Athens\UGA MSAI Program Material\3. Natural Language Processing\Project\Code and Filtered Data\count_vect_model.pkl", "wb")) #Store count_vec model
# pickle.dump(tf_idf_model, open(r"C:\Users\PRAVEEN\Desktop\MSAI\1. University Of Georgia, Athens\UGA MSAI Program Material\3. Natural Language Processing\Project\Code and Filtered Data\tf_idf_model.pkl", "wb")) #Store TFIDF model


# ### Testing the Method

# In[30]:


# def rank_sentences(test_article_str, test_article_matrix, feature_names, top_n=3):
#     sents = nltk.sent_tokenize(test_article_str)
#     sentences = [nltk.word_tokenize(sent) for sent in sents]
#     tfidf_sent = [[test_article_matrix[feature_names.index(w.lower())]
#                    for w in sent if w.lower() in feature_names]
#                  for sent in sentences]
#     doc_val = sum(test_article_matrix)
#     sent_values = [sum(sent) / doc_val for sent in tfidf_sent]
#     ranked_sents = [pair for pair in zip(range(len(sent_values)), sent_values)]
#     ranked_sents = sorted(ranked_sents, key=lambda x: x[1] *-1)
#     return ranked_sents[:top_n]

# def tf_idf_summarizer(test_article_list, original_articles, topn): # or take one argument and clean inside the function
#     # article_merged
#     test_article_str = ". ".join(test_article_list) # convert the text from list of strings to a single piece of text
#     #arti(input)
#     original_articles_str = ". ".join(original_articles)
    
#     # Cleaning
# #     original_articles, cleaned_articles, cleaned_articles_merged = data_preprocess([article])
# #     cleaned_articles = cleaned_articles[-1]
# #     original_articles = original_articles[-1]
    
#     feature_names = count_vect_model.get_feature_names()
# # Get the dense tf-idf matrix for the document
#     test_article_term_matrix  = count_vect_model.transform(test_article_list) #cleaned
#     test_article_tfidf_matrix = tfidf.transform(test_article_term_matrix)
#     test_article_dense  = test_article_tfidf_matrix.todense()
#     test_article_matrix = test_article_dense.tolist()[0]
# #Writing summary
#     top_sents = rank_sentences(test_article_str, test_article_matrix, feature_names,top_n=n)
#     top_sents=sorted(top_sents, key=itemgetter(0))
#     summary = '.'.join([original_articles_str.split('.')[i]
#                     for i in [pair[0] for pair in top_sents]])
#     summary = ' '.join(summary.split())
#     return summary


# In[31]:


# tf_idf_summarizer(cleaned_articles[24998], original_articles[24998], 5)


# In[ ]:





# In[41]:


def rank_sentences(test_article_str, test_article_matrix, feature_names, top_n=3):
    sents = nltk.sent_tokenize(test_article_str)
    sentences = [nltk.word_tokenize(sent) for sent in sents]
    tfidf_sent = [[test_article_matrix[feature_names.index(w.lower())]
                   for w in sent if w.lower() in feature_names]
                 for sent in sentences]
    doc_val = sum(test_article_matrix)
    sent_values = [sum(sent) / doc_val for sent in tfidf_sent]
    ranked_sents = [pair for pair in zip(range(len(sent_values)), sent_values)]
    ranked_sents = sorted(ranked_sents, key=lambda x: x[1] *-1)
    return ranked_sents[:top_n]

def tfidf_transform(article, topn=5): # or take one argument and clean inside the function
    # Cleaning
    original_articles, cleaned_articles, cleaned_articles_merged = data_preprocess([article])
    test_article_str = ". ".join(cleaned_articles[-1])
    original_articles_str = ". ".join(original_articles[-1])
    
    feature_names = count_vect_model.get_feature_names()
# Get the dense tf-idf matrix for the document
    test_article_term_matrix  = count_vect_model.transform(cleaned_articles[-1]) #cleaned
    test_article_tfidf_matrix = tfidf_model.transform(test_article_term_matrix)
    test_article_dense  = test_article_tfidf_matrix.todense()
    test_article_matrix = test_article_dense.tolist()[0]
#Writing summary
    top_sents = rank_sentences(test_article_str, test_article_matrix, feature_names, top_n = topn)
    top_sents=sorted(top_sents, key=itemgetter(0))
    summary = [original_articles[-1][i[0]] for i in top_sents]
#     summary = '.'.join([original_articles_str.split('.')[i] for i in [pair[0] for pair in top_sents]])
#     summary = ' '.join(summary.split())
    return summary


# In[ ]:





# In[42]:


# tfidf_transform(data_train['article'][24998])


# In[ ]:





# In[ ]:




