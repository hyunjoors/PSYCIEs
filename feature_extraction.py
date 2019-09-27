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


class FeatureSelection:

    def __init__(self, input_data):
        self.input_data = input_data

    ###################################################################################################################
    # Bigram Bag-of-word
    # Representation of the text as a simple bag of words. Using a bigram (continuous sequence of 2 words) model to
    # maintain word order to a certain extent.
    # REFER: Team Procrustination
    ###################################################################################################################
    def bag_of_word(self, *param):
        vectorizer = TfidfVectorizer(param)
        bag_of_word_matrix = vectorizer.fit_transform(self.input_data)

        return bag_of_word_matrix

    ###################################################################################################################
    # doc2vec
    # A vector representation of documents based on word2vec. Useful in detecting the relationship between words by
    # using vectors.
    # REFER: Team Procrustination
    ###################################################################################################################
    def doc2vec(self, *param, vector_size):
        model = Doc2Vec(param)
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
    # REFER: R
    ###################################################################################################################
    def dtm(self, *param):
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
    ###################################################################################################################
    def sentiment_analysis(self, *param):
        
        return sentiment_analysis_matrix





    ###################################################################################################################
    # ELMo (Embeddings from Language Models)
    # ELMo is a deep contextualized word representation of text documents. It represents each word in a document
    # according to its context within the entire document, while implementing a neural-network.
    ###################################################################################################################
    def ELMo(self, *param):
        
        return ELMo_matrix





    ###################################################################################################################
    # Lexical Diversity
    # Lexical diversity is calculated using documents’ multiple indices, which are calculated as the ratio between the
    # number of types of tokens and number of tokens.
    ###################################################################################################################
    def lexical_diversity(self, *param):
        
        return lexical_diversity_matrix





    ###################################################################################################################
    # Readability Indices
    # Readability Indices are different measures of how difficult a text is to read. It is estimated by measuring a
    # text’s complexity. Complexity is measured using attributes such as word length, sentence lengths, and syllable
    # counts.
    ###################################################################################################################
    def readability(self, *param):
        
        return readability_matrix





    ###################################################################################################################
    # Topic Modeling
    # A text mining tool used to find semantic structure in a body of text to find the different topics in a collection
    # of documents.
    ###################################################################################################################
    def topic_modeling(self, *param):
        
        return topic_modeling_matrix