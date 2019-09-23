# September 20, 2019
# Here are the features that needs to be implemented. Please refer how other teams used/extracted features from the 
# data. Also, these are not the only functions that are used in the project. For example, doc2vec includes several
# other helper functions (e.g., lemma, label_sentences). Those helper functions are already implemented in each team's
# code, that feel free to copy&paste them here.
# Lastly, I have implemented bag-of-word to show what should be returned from a function. input_data will be
# train_data. Please get familiarize with all features before start coding.
# Also, Please keep the documenation (120 char fro each line) and MUST comment&describe what you did.
# Happy coding!

from nltk.stem import WordNetLemmatizer
import os
import sys
import pandas as pd
import numpy as np
import re





#######################################################################################################################
# Helper functions
# These functions are not the major feature extracting functions. (e.g., clean_text, lemma below)
#######################################################################################################################
# Pre-processing (not removing stopwords, no lemma, etc.)
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # only keep english words/letters
    words = text.lower().split()  # lowercase
    return ' '.join(words)

# Further clean text using wordnetlemmatizer
lem = WordNetLemmatizer()
def lemma(text):
    words = nltk.word_tokenize(text)
    return ' '.join([lem.lemmatize(w) for w in words])





class feature_selection:

    def __init__(self):
        pass





    ###################################################################################################################
    # Bigram Bag-of-word
    # Representation of the text as a simple bag of words. Using a bigram (continuous sequence of 2 words) model to
    # maintain word order to a certain extent.
    # REFER: Team Procrustination
    ###################################################################################################################
    def bag_of_word(self, input_data, param):
        vectorizer = TfidfVectorizer(param)
        bag_of_word_matrix = vectorizer.fit_transform(input_data)

        return bag_of_word_matrix





    ###################################################################################################################
    # doc2vec
    # A vector representation of documents based on word2vec. Useful in detecting the relationship between words by
    # using vectors.
    # REFER: Team Procrustination
    ###################################################################################################################
    def doc2vec(self, input_data, param):
        
        return doc2vec_matrix





    ###################################################################################################################
    # DTM (Document Term Matrix)
    # Count the frequency of each token (word) that occur in a collection or individual document.
    # REFER: R
    ###################################################################################################################
    def dtm(self, input_data, param):
        
        return dtmd_matrix





    ###################################################################################################################
    # Sentiment Analysis
    # Sentiment Analysis is used to extract and identify subjective information related to the emotion, such as
    # negation, amplification, profanity, joy, fear and surprise, behind the text response.
    ###################################################################################################################
    def sentiment_analysis(self, input_data, param):
        
        return sentiment_analysis_matrix





    ###################################################################################################################
    # ELMo (Embeddings from Language Models)
    # ELMo is a deep contextualized word representation of text documents. It represents each word in a document
    # according to its context within the entire document, while implementing a neural-network.
    ###################################################################################################################
    def ELMo(self, input_data, param):
        
        return ELMo_matrix





    ###################################################################################################################
    # Lexical Diversity
    # Lexical diversity is calculated using documents’ multiple indices, which are calculated as the ratio between the
    # number of types of tokens and number of tokens.
    ###################################################################################################################
    def lexical_diversity(self, input_data, param):
        
        return lexical_diversity_matrix





    ###################################################################################################################
    # Readability Indices
    # Readability Indices are different measures of how difficult a text is to read. It is estimated by measuring a
    # text’s complexity. Complexity is measured using attributes such as word length, sentence lengths, and syllable
    # counts.
    ###################################################################################################################
    def readability(self, input_data, param):
        
        return readability_matrix





    ###################################################################################################################
    # Topic Modeling
    # A text mining tool used to find semantic structure in a body of text to find the different topics in a collection
    # of documents.
    ###################################################################################################################
    def topic_modeling(self, input_data, param):
        
        return topic_modeling_matrix