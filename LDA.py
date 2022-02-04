#!/usr/bin/env python
# coding: utf-8

# In[10]:


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
lda_model = pickle.load(open(r"Data/lda_model_24.pkl", "rb"))
dictionary = pickle.load(open(r"Data/dictionary.pkl", "rb"))
# corpus = pickle.load(open(r"Data\\corpus.pkl", "rb"))
topic_dict = pickle.load(open(r"Data/topic_dict_24.pkl", "rb"))


# In[27]:


# for article in data_train[-1:]['article']:
#     articles = article


# In[28]:


# articles = "collection rare color photo berlin 1937 take thomas neumann uncover norwegian archive life german capital tumultuous decade capture scene vibrant city iron grip adolf hitler reich height power year later city ruin russian ally occupy victory time image take hitler berlin vibrant hitler take power collapse democratic weimar republic 1933 severe economic problem cause great depression drive ordinary german far right partys arm chilling picture building emblazon swastika scene ordinary life german business child sun drench square smile friend train station cart selling banana food vendor sunny park regal rare color photo berlin 1937 unique perspective capital pre war period stadtschloss berlin city palace heavily damage bombing demolish east german authority war devastated stadtschloss gutte allied bomb tear east german authority war currently rebuild ominous 1937 hitler peak power ordinary german content opposition ruthlessly crush smile unknown trio train station likely friend colleague photographer rally soldier civilians rally decorate streets berlin photo believe take labour day 1 1937 bustle cart sell fruit busy berlin street norwegian engineer thomas neumann 1901 1978 take photo work germany film kind similar image preserve norwegian collection colour picture give historian valuable view interwar period 2007 photo gallery give national archives norway daughter thomas neumann train electrical engineer dresden graduate 1928 work berlin 1933 neumann member national unity party fascist organisation appoint propaganda leader oslo akershus leave party 1937 october 1944 arrest illegal activity send notorious grini concentration camp echo history street scene show augustiner keller beer cellar central berlin building festoon nazi regalia power hitler consolidate power mid 1930 thank widespread disillusionment weimar republic youth little boy outside unknown sunny square berlin order intimidate picture show troop lining boulevard festoon swastikas anticipation parade relaxation berliner enjoy snack sun soak park crowd picture take thomas neumann norwegian engineer work germany church state swastika maypole outside berlin cathedral dock man suit aboard steamer preussen presumably approach germany quiet moment driver lean state car enjoy cigarette photo candid moment berliner candid picture show brownshirt member hitler paramilitary force lounge state car building drape malign nazi symbol 30 january 1933 president hindenburg pressure franz von papen appoint hitler chancellor germany shortly fuhrer seize power  nazi government restore prosperity end mass unemployment heavy military spend free market economy extensive public work undertake include construction autobahns boost employment year later germany invade poland begin destructive war world see estimate 60 million people lose life result second world war global political landscape change forever ripple 1937 day celebration celebration 700 year berlin history grand messe berlin situate berlin westend complete 1937 heavily bomb allied aircraft masse lkarge crowd berlin presumably connection labour day force picture military personnel father beneath decoration officer appear inspect man overseer guard pristine white uniform look gather crowd civilian walker mystery german street year later fill russian british american troop serene unknown park berlin heat summer 1937 mean sprinkler require grass verdant history flag snap flap breeze throng german celebrate day colourful berliner gather look giant maypole outside berlin city cathedral"
# original_articles, cleaned_articles, cleaned_articles_merged = data_preprocess([articles])


# In[11]:


# # Load stored data
# data_train = pd.read_csv(r"Data\\cleaned_training_data.csv")
# original_articles, cleaned_articles, cleaned_articles_merged = pickle.load(open(r"Data\\cleaned_training_data.pkl", "rb"))


# In[8]:


# n = 5000
# data_train = data_train[:n]
# original_articles = original_articles[:n]
# cleaned_articles = cleaned_articles[:n]
# cleaned_articles_merged = cleaned_articles_merged[:n]


# In[9]:


# print("data_train:",len(data_train))
# print("original_articles:",len(original_articles))
# print("cleaned_articles:",len(cleaned_articles))
# print("cleaned_articles_merged:",len(cleaned_articles_merged))


# In[10]:


# texts = [[word for word in article.split(" ")] for article in cleaned_articles_merged[:-1]]


# In[6]:


# dictionary = corpora.Dictionary(texts)
# # print(dictionary.id2token) ## to see the actual dictionary generated
# corpus = [dictionary.doc2bow(text) for text in texts]


# In[7]:


# # fit LDA model
# lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=24)


# In[5]:


# # Create Topic-Word dictionary
# topic_dict = {}
# for topic in range(0, lda_model.num_topics):
#     temp = {}
#     for token, score in lda_model.show_topic(topic, topn=1000):
#         if token.isalpha():
#             topic_dict[str(topic)+"_"+token] = score


# In[9]:


# # Store model, dictionary, corpus and topic dict
# pickle.dump(lda_model, open(r"Data\\lda_model_24.pkl", "wb"))
# pickle.dump(dictionary, open(r"Data\\dictionary.pkl", "wb"))
# pickle.dump(corpus, open(r"Data\\corpus.pkl", "wb"))
# pickle.dump(topic_dict, open(r"Data\\topic_dict_24.pkl", "wb"))
# pickle.dump(texts, open(r"Data\\texts.pkl", "wb"))


# In[10]:


# # Load models
# lda_model = pickle.load(open(r"Data\\lda_model_24.pkl", "rb"))
# dictionary = pickle.load(open(r"Data\\dictionary.pkl", "rb"))
# corpus = pickle.load(open(r"Data\\corpus.pkl", "rb"))
# topic_dict = pickle.load(open(r"Data\\topic_dict_24.pkl", "rb"))
# texts = pickle.load(open(r"Data\\texts.pkl", "rb"))


# #### Sentence ranking

# In[2]:


def create_new_weights(topic_dict, vector):
    weights_dict = {}
#     for topic_word, score in tqdm_notebook(topic_dict.items()):
    for topic_word, score in topic_dict.items():
        topic, word = topic_word.split("_")
        if word not in weights_dict:
            weights_dict[word] = score*vector[0][int(topic)][1]
        else:
            weights_dict[word] = weights_dict[word]+(score*vector[0][int(topic)][1])
    return weights_dict


# In[3]:


def score_sentences(original_articles, cleaned_articles, weights_dict):
    sen_score = {}
    for idx, sen in enumerate(cleaned_articles):
        if len(sen) > 3:
            score = 0
            for word in sen.split(" "):
                if word in weights_dict:
                    score += weights_dict[word]
#             print(sen)
#             print(original_articles)
#             print(idx)
            try:
                sen_score[original_articles[idx]] = score
            except:
                pass
    sen_score = sorted(sen_score, key=sen_score.get, reverse=True)
    return sen_score


# In[4]:


def lda_transform(article, topn=5):
    # Cleaning
    original_articles, cleaned_articles, cleaned_articles_merged = data_preprocess([article])
    cleaned_articles = cleaned_articles[-1]
    original_articles = original_articles[-1]
    
    # Create a new corpus, made of previously unseen documents.
    texts_new = [[word for word in article.split(" ")] for article in cleaned_articles_merged]
    other_corpus = [dictionary.doc2bow(text) for text in texts_new]
    vector = lda_model.get_document_topics(other_corpus, minimum_probability=0.0)
    weights_dict = create_new_weights(topic_dict, vector)
    sen_score = score_sentences(original_articles, cleaned_articles, weights_dict)
    ids = sorted([original_articles.index(sen) for sen in sen_score[:topn] if sen in original_articles])
    extracted_sens = [original_articles[i] for i in ids]
    return extracted_sens


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[12]:


# for article in data_train[-1:]['article']:
#     articles = article
# articles


# In[13]:


# summary = lda_transform(articles, topn=5)
# summary


# In[ ]:





# In[ ]:




