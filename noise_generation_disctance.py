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
import sklearn
from scipy.spatial import distance
from numpy import dot
from numpy.linalg import norm
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore", category = DeprecationWarning)

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

def update_model(mail_id):
    model  = gensim.models.ldamodel.LdaModel.load("model_file1") 
    email_df = pd.read_csv('emails_processed.csv')
    new_df = email_df[email_df['to'] == mail_id]
    final_df = email_df[email_df['to'].isin(new_df['to']) == False ]
    stop_words = stopwords.words('english')
    stop_words.extend(['from','to','re','fwd','edu','use'])
    data = new_df.body.values.tolist()
    data_words = list(sent_to_words(data))
    bigram = Phrases(data_words, min_count = 50, threshold = 50)
    trigram = Phrases(bigram[data_words], threshold = 50)
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

def add_noise(sm_scale,p_matrix,epsilon):
    print("Adding Sensitivity Sampler noise")
    noise = np.random.normal(loc = 0, scale=sm_scale,size = (10,10))
    n_p_matrix = p_matrix+noise
    return n_p_matrix

def normalised_kendall_tau_distance(values1, values2):
    """Compute the Kendall tau distance."""
    n = len(values1)
    assert len(values2) == n, "Both lists have to be of equal length"
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = values1
    b = values2  
    ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
    return ndisordered / (n * (n - 1))

def main():
    m = gensim.models.ldamodel.LdaModel.load("model_file1")
    model = m.show_topics(num_words = 10,formatted = False)
    topics = get_topic_word_list(model)
    w_m = get_word_matrix(topics)
    p_m = get_prob_matrix(topics)
    # smooth sensitivity, epsilon for total privacy budget, 10 by 10 matrix
    epsilon = 5
    alpha = 11
    sm_sensitivity = 0.0421
    epsilon_R = (epsilon-math.log(1/0.001)/(alpha-1))/10/10
    #epsilon_T = (epsilon)/100
    sm_scale = alpha*sm_sensitivity*sm_sensitivity/epsilon_R/2
    #Adding Sensitivity Sampler noise
    print("Sensitivity sampler\n")
    n_p_m = add_noise(sm_scale,p_m,epsilon)
    
    print("L1 distance for Gaussian:\n")
    l1 = 0
    for row in range(len(n_p_m)):
        euc = distance.euclidean(p_m[row], n_p_m[row])
        l1 = l1+euc
    l1 = l1/len(n_p_m)
    print(l1)

    print("Kendall's tau distance for Gaussian:\n")
    k1 = 0
    for row in range(len(n_p_m)):
        ken = normalised_kendall_tau_distance(p_m[row], n_p_m[row])
        k1 = k1+ken
    k1 = k1/len(n_p_m)
    print(k1)

    rmse=0
    print('RMSE for Gaussian:\n')
    for row in range(len(n_p_m)):
        mse = sklearn.metrics.mean_squared_error(p_m[row], n_p_m[row])
        rmse = rmse+math.sqrt(mse)
    rmse = rmse/len(n_p_m)
    print(rmse)

if __name__=='__main__':
    main()
