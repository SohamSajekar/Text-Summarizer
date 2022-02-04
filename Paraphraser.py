#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import tqdm
from tqdm.notebook import tqdm_notebook
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from Cleaning import data_preprocess
import pickle
nlp = spacy.load('en_core_web_md')

# Getting simillar words using glove
import os
import urllib.request
import matplotlib.pyplot as plt
from scipy import spatial
from sklearn.manifold import TSNE
import numpy as np

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
stop_words=set(stopwords.words('English'))

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
stop_words=set(stopwords.words('English'))


# In[2]:


# # global emmbed_dict, quant_list
# ## List of mesurable quantities and its units
# quant_list = ['length', 'area', 'volume', 'weight', 'data', 'speed', 'acres', 'ares', 'hectares', 'square', 'feet', 'fts', 'ft', 'inchs', 'inch', 'inches', 'yards', 'yard', 'yd', 'miles', 'mile', 'mils', 'fahrenheit', 'gallons', 'gallon', 'litres', 'millilitres', 'cubic', 'tons', 'ton', 'pounds', 'bit', 'bits', 'byte', 'bytes', 'kilobyte', 'kilobytes', 'megabyte', 'megabytes', 'gigabyte', 'gigabytes', 'terabyte', 'terabytes', 'per', 'mass', 'time', 'temperature', 'electric', 'current', 'second', 'seconds', 'sec', ' minute', 'min', 'minutes', 'hour', 'hours', 'hrs', 'day', 'days', 'week', 'weeks', 'year', 'years', 'decade', 'decades', 'century', 'centuries', 'millimeters', 'mm', 'millimetre', 'centimeters', 'cm', 'meters,m', 'mtrs', 'metre and kilometers', 'km', 'kilogram', 'kg', 'ounce', 'oz', 'pound', 'lbs', 'gram', 'gm', 'degrees', 'celsius', 'C', 'kelvin', 'K']
# emmbed_dict = {}
# with open('/Users/sanikakatekar/Downloads/Jupyter Notebooks/glove.6B.200d.txt','r') as f:
#     for line in f:
#         values = line.split()
#         word = values[0]
#         vector = np.asarray(values[1:],'float32')
#         emmbed_dict[word]=vector


# In[3]:


# pickle.dump(quant_list, open("measurable_quantities.pkl", "wb"))
# pickle.dump(emmbed_dict, open("glove200d.pkl", "wb"))


# In[4]:


quant_list = pickle.load(open("Data/measurable_quantities.pkl", "rb"))
emmbed_dict = pickle.load(open("Data/glove200d.pkl", "rb"))


# In[5]:


def find_similar_word(main_word, topn=50):
    nearest = sorted(emmbed_dict.keys(), key=lambda word: spatial.distance.cosine(emmbed_dict[word], main_word))
    return nearest[1:topn]


# In[6]:


def get_best_synonym(bigram, sen_pos):
    original_word, main_tag = sen_pos[bigram[0]]
    syn = original_word
#     if (main_tag in ['NN','CD','RB','MD','VBN','VBD','NNP','NNPS']) or:
#         return original_word
    score = 100
    original_word = original_word.lower()
    context_word_1 = sen_pos[bigram[1]][0].lower()
    context_word_2 = sen_pos[bigram[2]][0].lower()
    try:
        vec_org_wrd = emmbed_dict[original_word]
        vec_context = emmbed_dict[context_word_1] + emmbed_dict[context_word_2]
    except:
        return syn

#     def get_lowest(wrd, score):
#         vec_1 = vec_org_wrd + vec_context
#         vec_2 = emmbed_dict[wrd] + vec_context
#         bigram_cosine = spatial.distance.cosine(vec_1, vec_2)
#         if bigram_cosine < score:
#             score = bigram_cosine
#             syn = wrd
#         return syn, score
    
    for wrd in find_similar_word(vec_org_wrd, topn=50):
        syn_tag = nltk.tag.pos_tag([wrd])[0][1]
        if main_tag == syn_tag:
            vec_1 = vec_org_wrd + vec_context
            vec_2 = emmbed_dict[wrd] + vec_context
            bigram_cosine = spatial.distance.cosine(vec_1, vec_2)
            if bigram_cosine < score:
                score = bigram_cosine
                syn = wrd
#             syn, score = get_lowest(wrd, score)
        elif (main_tag == 'CC') and (syn_tag == 'IN'):
            vec_1 = vec_org_wrd + vec_context
            vec_2 = emmbed_dict[wrd] + vec_context
            bigram_cosine = spatial.distance.cosine(vec_1, vec_2)
            if bigram_cosine < score:
                score = bigram_cosine
                syn = wrd
#             syn, score = get_lowest(wrd, score)
    return syn


# In[16]:


def rephrase_sen(sens):
    summary = []
#     for sen in tqdm_notebook(sens):
    for sen in sens:
        new_sen = ""
        ## Basic cleaning
#         original_articles, cleaned_articles, cleaned_articles_merged = data_preprocess([sen])
#         cleaned_articles_merged  = cleaned_articles_merged[-1]
#         cleaned_articles = cleaned_articles[-1]
#         original_articles = original_articles[-1]
        
        words = word_tokenize(sen)
        sen_pos = nltk.tag.pos_tag(words)
        
        sen_entity = [X.text for X in nlp(sen).ents] + quant_list
#         for wrd in tqdm_notebook(words):
        for wrd in words:
            if (wrd in sen_entity) or (wrd.lower() in stop_words):
                new_sen += wrd + " "
            elif wrd.isalpha():
                for idx, pos in enumerate(sen_pos):
                    word, tag = pos
                    if (word == wrd) and (tag not in ['NN','CD','RB','MD','VBN','VBD','NNP','NNPS']):
                        if idx == 0:
                            ids = (idx, idx+1, idx+2)
                        elif ((idx+1)==len(sen_pos)):
                            ids = (idx, idx-1, idx-2)
                        else:
                            ids = (idx, idx-1, idx+1)
                        synonym = get_best_synonym(ids, sen_pos)
                        new_sen += "\033[1m\033[94m"+synonym+"\033[0m" + " "
                    elif (word == wrd):
                        new_sen += wrd + " "    
            else:
                new_sen += wrd + " "
        summary.append(new_sen)
    summary = [sen.strip() + "." for sen in summary]
    return " ".join(summary)


# In[17]:


# extracted_sen = ["But at the time these images were taken, Hitler's Berlin was vibrant", 
#                  "Just two years later Germany would invade Poland and begin the most destructive war the world has ever seen", 
#                  "An estimated 60 million people lost their lives as a result of the Second World War and the global political landscape changed forever"]


# In[18]:


# rephrase_sen(extracted_sen)


# In[ ]:




