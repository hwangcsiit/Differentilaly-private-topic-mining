import numpy as np
import random 
import statistics
#!/usr/bin/env python
# -*- coding: utf-8 -*-

#from __future__ import print_function

import logging
from scipy.special import lambertw
import gensim.models
from gensim.test.utils import datapath
import pandas as pd
import numpy as np
import math
import re
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser
import spacy
import nltk
from numpy import dot
from numpy.linalg import norm
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore", category = DeprecationWarning)
import csv

def get_word_matrix(topic_word_list):
    p_list = []
    temp_list = []
    for words in topic_word_list:
        for w in range(len(words)):
            temp_list.append(words[w][0])
        p_list.append(temp_list)
        temp_list = []
    matrix_of_words = np.array(p_list)
    return matrix_of_words

def get_prob_matrix(topic_word_list):
    pb_list = []
    temp_list = []
    for words in topic_word_list:
        for w in range(len(words)):
            temp_list.append(words[w][1])
        pb_list.append(temp_list)
        temp_list = []
    matrix_of_probability = np.array(pb_list)
    return matrix_of_probability

def get_topic_word_list(m):
    topic_word_list = []
    for topic in m:
        topic_word_list.append(sorted(topic[1]))
    return topic_word_list

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc = True))

def remove_stopwords(texts):
    stop_words = stopwords.words('english')
    stop_words.extend(['from','to','re','fwd','edu','use'])
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts, trigram_mod, bigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags = ['NOUN','ADJ','VERB','ADV']):
    nlp = spacy.load('en_core_web_sm', disable = ['parser','ner'])
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def update_model(final_df):
    model  = gensim.models.ldamodel.LdaModel.load("model_file1") 
    stop_words = stopwords.words('english')
    stop_words.extend(['from','to','re','fwd','edu','use'])
    data = final_df.body.values.tolist()
    data_words = list(sent_to_words(data))
    bigram = Phrases(data_words, min_count = 500, threshold = 500)
    trigram = Phrases(bigram[data_words], threshold = 500)
    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)
    data_words_nostops = remove_stopwords(data_words)
    data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)
    data_words_trigrams = make_trigrams(data_words_bigrams, trigram_mod, bigram_mod)
    nlp = spacy.load('en_core_web_sm', disable = ['parser','ner'])
    data_lemmatized = lemmatization(data_words_trigrams, allowed_postags = ['NOUN','ADJ','VERB','ADV'])
    id2word = corpora.Dictionary(data_lemmatized)
    u_corpus = [id2word.doc2bow(text) for text in data_lemmatized]
    model.update(u_corpus)
    return model

def get_user_list(data):
    email_df = data
    to_list = email_df['to']
    unique_list = np.unique(to_list)
    print(len(unique_list))
    return unique_list
  
def sensitivity_calculation(all_prob, diff_prob):
    out_mat = np.sqrt(pow(all_prob-diff_prob , 2))
    #print(out_mat)
    out = np.linalg.norm(out_mat)
    return out

def create_new_row(row1, row2):
    out_row = []
    counter=0
    while counter < len(row1):
        if row1[counter]==row2[counter]:
            out_row.append(row1[counter])
        else:
            out_row.append(row1[counter])
            out_row.append(row2[counter])
        counter = counter+1
    return out_row

def update_probability_matrix(w_matrix,w1,p1):
    row=0
    p_matrix = []
    while row < len(w_matrix):
        ele=0
        w1_ele=0
        temp_row = []
        while ele < len(w_matrix[0]):
            if w1_ele<len(w1[0]) and w_matrix[row][ele] == w1[row][w1_ele]:
                temp_row.append(p1[row][w1_ele])
                ele = ele+1
                w1_ele = w1_ele+1
            else:
                temp_row.append(0.0)
                ele = ele+1
        row = row+1
        p_matrix.append(temp_row)
    return p_matrix


def sensitivityWhole(email_df1,email_df2):
    m = update_model(email_df1)
    model = m.show_topics(num_words = 10,formatted = False)
    topics = get_topic_word_list(model)
    w_m = get_word_matrix(topics)
    p_m = get_prob_matrix(topics)

    new_m = update_model(email_df2)
    new_model = new_m.show_topics(num_words = 10,formatted = False)
    new_topics = get_topic_word_list(new_model)
    new_w_m = get_word_matrix(new_topics)
    new_p_m = get_prob_matrix(new_topics)
    new_w_matrix = []
    temp_row = []
    p_temp_row = []
    row_num = 0
    while row_num < len(w_m):
        temp_row = create_new_row(w_m[row_num], new_w_m[row_num])
        row_num = row_num+1
        new_w_matrix.append(temp_row)
    up_p_m = np.array(update_probability_matrix(w_m,new_w_m,p_m))
    up_new_p_m = np.array(update_probability_matrix(w_m,new_w_m,new_p_m))
    sensitivity = sensitivity_calculation(up_p_m,up_new_p_m)
    return sensitivity

def optimize_m_k(gamma):
	rho = math.exp(lambertw(-gamma / (2 * math.exp(0.5)),-1) + 0.5)
	m = math.log(1/rho)/(2*(gamma-rho)**2)
	k = (m * (1 - gamma + rho + math.sqrt(math.log(1/rho)/(2 * m))))
	return (math.ceil(m), int(k))

#P distribution is uniform distribution
def Sample_Sensitivity(database, user_list, n, m, k):
    GS = [] 
    for index1 in range(m):
        db1_user = list(np.random.choice(user_list, n-1))
        db2_user = db1_user.copy()
        db1_user.append(np.random.choice(user_list, 1)[0])
        db2_user.append(np.random.choice(user_list, 1)[0])
        db1 = pd.DataFrame(columns=('index','body', 'to', 'from'))
        db2 = pd.DataFrame(columns=('index','body', 'to', 'from'))

        for index in range (len(db1_user)):
            temp1 = []
            temp2 = []
            temp1.append(db1_user[index])  
            temp2.append(db2_user[index])          
            db1 = db1.append([database['to'].isin(temp1)])
            db2 = db2.append([database['to'].isin(temp2)])                
        GS.append(sensitivityWhole(db1, db2))
    GS = sorted(GS)
    sen = GS[k]
    return sen

#first and two parameters: database and size
gammalist = [0.1, 0.2, 0.3, 0.4, 0.5]
result = []
for i in range(5):
    gamma = gammalist[i]
    m, k = optimize_m_k(gamma)
    email_df = pd.read_csv('emails_processed.csv', nrows=30000)
    email_df=email_df.drop_duplicates()
    user_list = get_user_list(email_df)
    result.append(Sample_Sensitivity(email_df,user_list, n=30000, m=m, k=k))
print(result)

