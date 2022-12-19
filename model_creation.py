import re
import gensim 
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser
import spacy
import nltk
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore", category = DeprecationWarning)
import pandas as pd
import numpy as np

# Read the processed data
email_df = pd.read_csv('emails_processed.csv')

stop_words = stopwords.words('english')
stop_words.extend(['from','to','re','fwd','edu','use'])
data = email_df.body.values.tolist()
print('data length',len(data))

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc = True))

def remove_stopwords(texts):
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
def word_list_from_model(m):
    topic_word_list = []
    for topic in m:
        topic_word_list.append(topic[1])
    return topic_word_list

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

def main():
    stop_words = stopwords.words('english')
    stop_words.extend(['from','to','re','fwd','edu','use'])
    data = email_df.body.values.tolist()
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
    corpus = [id2word.doc2bow(text) for text in data_lemmatized]
    print(len(data_lemmatized))
    print(len(corpus))
    print(id2word.token2id)
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word,
                                          num_topics = 10,
                                           random_state = 100,
                                           update_every = 1,
                                           chunksize=100,
                                           passes = 10,
                                           alpha = 'auto',
                                           per_word_topics = True)
    model = lda_model.show_topics(num_words = 10,formatted = False)
    lda_model.save("model_file10by10")
    
if __name__=='__main__':
    main()
