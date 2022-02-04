#!/usr/bin/env python
# coding: utf-8

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


# In[2]:


## Load Data
# data_train = pd.read_csv(r"Data\\cnn_dailymail\\train.csv")
# data_test = pd.read_csv(r"dataset\\cnn_dailymail\\test.csv")
# data_val = pd.read_csv(r"dataset\\cnn_dailymail\\validation.csv")


# In[3]:


# data_train = data_train.sample(n = 25000)
# data_train


# In[4]:


def clean_article(article):
    # Remove "(CNN) -- "
    index = article.find('(CNN) -- ')
    if index > -1:
        article = article[index+len('(CNN)'):]
    # Removing source information    
    article = re.sub(r'By\s\..*?\s\.', '', article)
    article = re.sub(r'PUBLISHED:\s\..*?\s\.', '', article)
    article = re.sub(r'UPDATED:\s\..*?\s\.', '', article)
    # Removing words with collen
    article = re.sub(r'([\w]+):', '', article)
    # Removing space before period "."
    article = re.sub(r'\s(\.)', r'\1', article)
    # Removing unwanted periods
    cleaned_article = re.sub(r'\.\s([a-z0-9])', r' \1', article)
   
    # Removing hypens 
    article = re.sub(r'-', r' ', cleaned_article)
    # Removing all punctuations except period and hypens
    article = re.sub(r'[^\w\s\.-]', '', article)
    # Removing multiple spaces in-between words
    article = re.sub(r'\s{1,}', r' ', article)
    return article, cleaned_article


# In[5]:


# Separating sentences 
def generate_sen(article):
    original_sentences = nltk.tokenize.sent_tokenize(article)
    original_sentences = [re.sub(r'\.', '', sen).strip() for sen in original_sentences if len(sen) > 2]
    # sentences = [sen.lower() for sen in sentences if len(sen) > 0]
    return original_sentences


# In[6]:


# Removing Stopwords and Lemmatization
def stopwords_n_lemma(original_sentences):
    cleaned_sentences = []
    for sen in original_sentences:
        new_sen = ""
        for word in sen.split():
            if word.lower() not in sw_spacy:
                new_sen += word.lower() + " "
        cleaned_sentences.append(" ".join([token.lemma_ for token in en(new_sen.strip())]))
    return cleaned_sentences


# In[7]:


def data_preprocess(articles):
    original_articles = []
    cleaned_articles = []
    cleaned_articles_merged = []
    for article in articles:
        article, cleaned_article = clean_article(article)
        cleaned_sentences = stopwords_n_lemma(generate_sen(article))
        original_articles.append(generate_sen(cleaned_article))
        cleaned_articles.append(cleaned_sentences)
        cleaned_articles_merged.append(" ".join(cleaned_sentences))
    return original_articles, cleaned_articles, cleaned_articles_merged


# In[8]:


# articles = data_train['article'].values
# original_articles, cleaned_articles, cleaned_articles_merged = data_preprocess(articles)


# In[17]:


# Store cleaned data
# cleaned_training_data = [new_original_articles, cleaned_articles, cleaned_articles_merged]
# pickle.dump(cleaned_training_data, open(r"Data\\cleaned_training_data.pkl", "wb"))


# In[33]:


# Save dataframe
# data_train.reset_index(drop=True).to_csv(r"Data\\cleaned_training_data.csv", index=False)

