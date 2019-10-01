# September 20, 2019
# Here are the features that need to be implemented. Please refer to how other teams used/extracted features from the
# data. Also, these are not the only functions that are used in the project. For example, doc2vec includes several
# other helper functions (e.g., lemma, label_sentences). Those helper functions are already implemented in each team's
# code, and feel free to copy and paste them here.
# Lastly, I have implemented bag-of-word to show what should be returned from a function. input_data will be
# train_data. Please get familiarized with all the features before starting to code.
# Also, Please keep the documentation (120 char for each line) and comment & describe what you have done.
# Happy coding!

from gensim.models import Doc2Vec
from gensim.models import doc2vec
from nltk.stem import WordNetLemmatizer
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


class FeatureSelection:

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
        vectorizer = TfidfVectorizer(self.bag_of_word_param)
        bag_of_word_matrix = vectorizer.fit_transform(self.input_data)

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
    # DTM (Document Term Matrix)
    # Count the frequency of each token (word) that occur in a collection or individual document.
    # REFER: PI-RATES (R)
    ###################################################################################################################
    def dtm(self):
        # Don't know the exact structure of data being sent as argument so depending on that can change this function

        # pipeline = [lambda s: re.sub('[^\w\s]', '', s),
        #             lambda s: re.sub('[\d]', '', s),
        #             lambda s: s.lower(),
        #             lambda s: ' '.join(map(stem, shorttext.utils.tokenize(s)))]  # Pre-processing pipeline
        # text_preprocessor = shorttext.utils.text_preprocessor(pipeline)
        # docids = [i for i in range(0, len(self.input_data))]
        # corpus = [text_preprocessor(response).split(" ") for response in self.input_data]
        # dtmd_matrix = shorttext.utils.DocumentTermMatrix(corpus, docids=docids, tfidf=False)

        docids = [i for i in range(0, len(self.input_data))]
        corpus = [response for response in self.input_data]
        dtmd_matrix = shorttext.utils.DocumentTermMatrix(corpus, docids=docids, tfidf=False)
        
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
    # ELMo (Embeddings from Language Models)
    # ELMo is a deep contextualized word representation of text documents. It represents each word in a document
    # according to its context within the entire document, while implementing a neural-network.
    # REFER: Natural Selection (Py)
    ###################################################################################################################
    def ELMo(self):
        
        return ELMo_matrix





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