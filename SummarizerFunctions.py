#!/usr/bin/env python
# coding: utf-8

# In[16]:


import re
import numpy as np
import pandas as pd
from LDA import lda_transform
from LSA import lsa_transform
from rouge_metric import PyRouge
from TFIDF import tfidf_transform
from Paraphraser import rephrase_sen


# In[17]:


def clean(line):
    # Removing space before period "."
    line = re.sub(r'\s(\.)', r'\1', line)
    # Replacing '\n' with white space
    line = line.replace(r"\n", " ")
    return line


# In[23]:


def get_rouge(summary, highlight):
    summary = [summary]
    highlight = [[highlight]]
    rouge = PyRouge(rouge_n=(1, 2), rouge_l=True, rouge_s=True, rouge_su=True, skip_gap=3)
    scores = rouge.evaluate(summary, highlight)
    rouge_dict = {}
    for k,v in scores.items():
        for k1,v1 in v.items():
            rouge_dict[k+" "+k1] = v1
    return rouge_dict


# In[24]:


def extract_summary(idx, arti, highlight, topn, method):
    print("*"*60)
    print("\033[1mProcessing Article", idx+1,"\033[0m")
    print(arti)
    
    print("\n\033[1mActual highlight\033[0m")
    print(highlight)

    print("\n\033[1mextracting best sentences... ", end='')
    if method == 'lda':
        extracted_sens = lda_transform(arti, topn)
    elif method == 'lsa':
        extracted_sens = lsa_transform(arti, topn)
    elif method == 'tfidf':
        extracted_sens = tfidf_transform(arti, topn)
    extractive_summary = " ".join([sen.strip() + "." for sen in extracted_sens])
    extractive_rouge = get_rouge(extractive_summary, highlight)
    print("completed.\033[0m")
    print(extractive_summary)
    print("\n\033[1mrouge scores for extracted summary:\033[0m")
    for k,v in extractive_rouge.items():
        print(k+":",v)
    print("\n")
    
    print("\033[1mparaphrasing extracted summary... ", end='')
    summary = rephrase_sen(extracted_sens)
    paraphrased_rouge = get_rouge(summary, highlight)
    print("completed.\033[0m")
    print(summary)
    print("\n\033[1mrouge scores for extracted summary:\033[0m")
    for k,v in paraphrased_rouge.items():
        print(k+":",v)
    print("*"*60+"\n\n")
#     return summary


# In[25]:


def get_summary(articles, highlights, topn=3, method='lsa'):
    if type(articles) == str:
        articles = [articles]
    if type(highlights) == str:
        highlights = [highlights]
    for idx, arti in enumerate(articles):
        extract_summary(idx, arti, highlights[idx], topn, method)


# In[ ]:





# In[ ]:





# In[26]:


# # Load Data
# test_data = pd.read_csv("Data/cnn_dailymail/test.csv")
# test_data = test_data.iloc[[10000]] # Total 11490 samples
# # test_data = test_data.sample(n=1).reset_index(drop=True) # sampling a few articles
# articles = test_data['article'].to_list() # creating list of dataframe column
# highlights = test_data['highlights'].to_list()
# highlights = [clean(sen) for sen in highlights] # cleaning sens: removing spaces before "."
# # articles


# In[27]:


# get_summary(articles, highlights, topn=3, method='lda')


# In[28]:


# get_summary(articles, highlights, topn=3, method='lsa')


# In[8]:


# get_summary(articles, highlights, topn=3, method='tfidf')


# In[ ]:




