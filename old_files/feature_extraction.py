#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Monday, ‎September ‎23, ‎2019, ‏‎12:47:06 PM
Last Modified on 2/11/2020
@author: Hyun Joo Shin

@Note:
GLoVe?
"""

from keras.layers import Dense, Dropout, Embedding, Flatten, Input, MaxPooling1D
from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasRegressor
from keras import backend as K 
import keras.layers as layers
from keras.models import Model, load_model
from keras.engine import Layer

from textblob import TextBlob
from gensim.models import Doc2Vec
from gensim.models import doc2vec
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from scipy import stats
from sklearn import utils
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import SparsePCA, TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from spellchecker import SpellChecker
from stemming.porter import stem
from textstat.textstat import textstatistics, easy_word_set, legacy_round 
from tqdm import tqdm
import language_check
import nltk
import numpy as np
import os
import pandas as pd
import re
import scipy
import shorttext
import spacy
import sys
import tensorflow_hub as hub
import tensorflow as tf



#######################################################################################################################
# Helper functions
# These functions are not the major feature extracting functions. (e.g., clean_text, lemma below)
#######################################################################################################################
# Further clean text using WordNetLemmatizer

lem = WordNetLemmatizer()
def lemma(text):
    words = nltk.word_tokenize(text)
    return ' '.join([lem.lemmatize(w) for w in words])

def syllables_count(text): 
    return textstatistics().syllable_count(text) 

def difficult_word_count(text):
    return textstatistics().difficult_words(text)

def sentence_count(text):
    return textstatistics().sentence_count(text)

def avg_syllables_per_word(text): 
    nsyllables=syllables_count(text)
    nwords=word_count(text)
    ASPW=float(nsyllables)/float(nwords)
    return legacy_round(ASPW,2)

def avg_sentence_length(text): 
    nwords = word_count(text) 
    nsentences = sentence_count(text) 
    average_sentence_length = float(nwords / nsentences) 
    return legacy_round(average_sentence_length,2)
  
def flesch_ease_score(text):
    return textstatistics().flesch_reading_ease(text)
    
def flesch_grade_score(text):
    return textstatistics().flesch_kincaid_grade(text)

def linsear_write_score(text):
    return textstatistics().linsear_write_formula(text)

def dale_chall_score(text):
    return textstatistics().dale_chall_readability_score(text)

def gunning_fog_score(text):
    return textstatistics().gunning_fog(text)

def smog_score(text):
    return textstatistics().smog_index(text)

def automated_readability_score(text):
    return textstatistics().automated_readability_index(text)

def coleman_liau_score(text):
    return textstatistics().coleman_liau_index(text)

def word_count(text): 
    return textstatistics().lexicon_count(text, removepunct=True)

def tokenize(text):
    return TextBlob(text).words

# Create a custom layer that allows us to update weights (lambda layers do not have trainable parameters!)
class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable=True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                      as_dict=True,
                      signature='default',
                      )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)

def ElmoRegressionModel(
    dense_dropout_rate=0.5,
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['mse'],
    print_summary=True,
    include_hidden_layer=False,
    hidden_layer_size=64
):
    inputs, embeddings = [], []
    
    for idx in range(1, 6):
        _input = layers.Input(shape=(1,), dtype="string")
        inputs.append(_input)
        embedding = ElmoEmbeddingLayer()(_input)
        embeddings.append(embedding)
        
    concat = layers.concatenate(embeddings)
    dense = Dropout(dense_dropout_rate)(concat)
    if include_hidden_layer:
        dense = layers.Dense(hidden_layer_size, activation='relu')(dense)
        dense = Dropout(dense_dropout_rate)(dense)
    dense = layers.Dense(1, activation='relu')(dense)# (drop2)
    
    # If we want to do 5-way prediction within a single network
    # dense = layers.Dense(5, activation='relu')(dense)
    
    model = Model(inputs=inputs, outputs=dense)

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    
    if print_summary:
        model.summary()

    return model


class FeatureExtraction:

    def __init__(self, input_data, features_dict):
        self.input_data = input_data
        self.bag_of_word_param = features_dict['bag_of_word']
        self.doc2vec_param = features_dict['doc2vec']
        self.dtm_param = features_dict['dtm']
        self.sentiment_analysis_param = features_dict['sentiment_analysis']
        self.ELMo_param = features_dict['ELMo']
        self.lexical_diversity_param = features_dict['lexical_diversity']
        self.readability_param = features_dict['readability']
        self.topic_modeling_param = features_dict['topic_modeling']

    ###################################################################################################################
    # Bigram Bag-of-word
    # Representation of the text as a simple bag of words. Using a bigram (continuous sequence of 2 words) model to
    # maintain word order to a certain extent.
    # REFER: Team Procrustination
    ###################################################################################################################
    def bag_of_word(self):
        # print(self.bag_of_word_param)
        vectorizer = TfidfVectorizer(self.bag_of_word_param)
        vector = vectorizer.fit_transform(self.input_data)
        
        # get the first vector out (for the first document)
        first_vector_tfidfvectorizer=vector[0]
        
        # place tf-idf values in a pandas data frame
        bag_of_word_matrix = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=vectorizer.get_feature_names(), columns=["tfidf"])
        bag_of_word_matrix.sort_values(by=["tfidf"],ascending=False)
        # print(bag_of_word_matrix)

        return bag_of_word_matrix

    ###################################################################################################################
    # doc2vec
    # A vector representation of documents based on word2vec. Useful in detecting the relationship between words by
    # using vectors.
    # REFER: Team Procrustination
    ###################################################################################################################
    def doc2vec(self):
        model = Doc2Vec(self.doc2vec_param)
        model.build_vocab([x for x in tqdm(self.input_data)])

        for epoch in range(10):  # Train the model for 10 epochs
            model.train(utils.shuffle([x for x in tqdm(self.input_data)]), total_examples=len(self.input_data),
                        epochs=1)  # Reshuffle
            model.alpha -= 0.002
            model.min_alpha = model.alpha

        doc2vec_matrix = np.zeros((model, vector_size))  # Number of dimensional feature vectors to be extracted
        for i in range(0, len(self.input_data)):
            prefix = str(i)  # Don't know if prefix is correct!?
            doc2vec_matrix[i] = model.docvecs[prefix]

        return doc2vec_matrix

    ###################################################################################################################
    # DTM (Document Term Matrix) ????Dynamic Topic Modeling
    # Count the frequency of each token (word) that occur in a collection or individual document.
    # REFER: PI-RATES (R)
    ###################################################################################################################
    def dtm(self):

        docids = [i for i in range(0, len(self.input_data))]
        corpus = [response for response in self.input_data]
        dtmd_matrix = shorttext.utils.DocumentTermMatrix(corpus, docids=docids, tfidf=False)
        
        
        pipeline = [lambda s: re.sub('[^\w\s]', '', s),
            lambda s: re.sub('[\d]', '', s),
            lambda s: s.lower(),
            lambda s: ' '.join(map(stem, shorttext.utils.tokenize(s)))
        ]
        txtpreproceesor = shorttext.utils.text_preprocessor(pipeline)
        docids = list(usprezdf['yrprez'])    # defining document IDs
        corpus = [txtpreproceesor(speech).split(' ') for speech in usprezdf['speech']]
        dtm = shorttext.utils.DocumentTermMatrix(corpus, docids=docids, tfidf=False)

        
        return dtmd_matrix

    ###################################################################################################################
    # Sentiment Analysis
    # Sentiment Analysis is used to extract and identify subjective information related to the emotion, such as
    # negation, amplification, profanity, joy, fear and surprise, behind the text response.
    # REFER: PI-RATES (R), Logistic Aggression (R)
    ###################################################################################################################
    def sentiment_analysis(self):
        
        return sentiment_analysis_matrix





    ###################################################################################################################
    # Lexical Diversity
    # Lexical diversity is calculated using documents’ multiple indices, which are calculated as the ratio between the
    # number of types of tokens and number of tokens.
    # REFER: Logistic Aggression (R)
    ###################################################################################################################
    def lexical_diversity(self):
        
        return lexical_diversity_matrix





    ###################################################################################################################
    # Readability Indices
    # Readability Indices are different measures of how difficult a text is to read. It is estimated by measuring a
    # text’s complexity. Complexity is measured using attributes such as word length, sentence lengths, and syllable
    # counts, speeling errors, profanity, etc.
    # REFER: Natural Selection (Py), PI-RATES (R), Logistic Aggression (R)
    ###################################################################################################################
    def readability(self):
        readability_matrix = pd.DataFrame()
        readability_matrix["syllables_count"] = self.input_data.apply(syllables_count) 
        readability_matrix["word_count"] = self.input_data.apply(word_count) 
        readability_matrix["difficult_count"] = self.input_data.apply(difficult_word_count) 
        readability_matrix["sentence_count"] = self.input_data.apply(sentence_count) 
        readability_matrix["avg_syllables_per_word"] = self.input_data.apply(avg_syllables_per_word)
        readability_matrix["avg_sentence_length"] = self.input_data.apply(avg_sentence_length) 
        readability_matrix["flesch_ease_score"] = self.input_data.apply(flesch_ease_score) 
        readability_matrix["flesch_grade_score"] = self.input_data.apply(flesch_grade_score) 
        readability_matrix["linsear_write_score"] = self.input_data.apply(linsear_write_score) 
        readability_matrix["dale_chall_score"] = self.input_data.apply(dale_chall_score) 
        readability_matrix["smog_score"] = self.input_data.apply(smog_score) 
        readability_matrix["coleman_liau_score"] = self.input_data.apply(coleman_liau_score)
        
        return readability_matrix





    ###################################################################################################################
    # Topic Modeling
    # A text mining tool used to find semantic structure in a body of text to find the different topics in a collection
    # of documents.
    # REFER: Logistic Aggression (R)
    ###################################################################################################################
    def topic_modeling(self):
        
        return topic_modeling_matrix





    ###################################################################################################################
    # ELMo (Embeddings from Language Models)
    # ELMo is a deep contextualized word representation of text documents. It represents each word in a document
    # according to its context within the entire document, while implementing a neural-network.
    # REFER: Natural Selection (Py)
    ###################################################################################################################
    def ELMo(self):
        
        return ELMo_matrix