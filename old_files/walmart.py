# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '..\\Algorithms\01_Natural_Selection'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# ## Natural Selection (Walmart) - SIOP 2019 Competition
# 
# - Natural Selection = Natural Language Processing + Global Selection and Assessment
# 
# This competition consisted of a data set containing open-ended resposes to 5 situational judgment items and 5 aggregated personality trait scores. The goal of the competition was to generate the best mean prediction across all 5 traits using only these open-ended responses.
# 
# We used three approaches:
# - Key Words: a sample of responses from the high- and low-end of each trait distribution were read and then key words were extracted which seemed to occur more at one end of the distribution than the other
# - Machine learning: machine learning techniques were used with features from Key Words and other data extracted from the text
# - Deep learning: deep learning techniques were used. This is the most refined code and the place where experienced data scientists would find most value in reviewing
# 
# The winning submission resulted from combining the methods.
# 
# Note on the code contained in this notebook:
# - We removed most of the exploratory code from this notebook to focus on what we actually used in the final predictions. Some irrelevant and duplicte elements remain. This code was written by different people with different levels of coding expertise. Thus, the application of code can vary widely and may seem disjointed/incoherent at times.
#%% [markdown]
# ## Dependencies
# 
# - pandas (https://pandas.pydata.org/)
# - numpy (http://www.numpy.org/)
# - seaborn (https://seaborn.pydata.org/)
# - scikit-learn (https://scikit-learn.org/)
# - scipy (https://www.scipy.org/)
# - pyspellchecker (https://github.com/barrust/pyspellchecker)
# - textblob (https://textblob.readthedocs.io/en/dev/)
# - spacy (https://spacy.io/)
# - tpot (http://epistasislab.github.io/tpot/)
# - xgboost (https://xgboost.readthedocs.io/en/latest/)

#%%
# Python's best-known DataFrame implementation
import pandas as pd

# Fast, flexible array and numerical linear algebra subroutines
import numpy as np

# OS utilities (e.g. path module)
import os

# Plots & other visualization
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

# Pretty printing of complex datatypes
from pprint import pprint
import json

# Preprocessing and modeling utilities
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import SparsePCA, TruncatedSVD
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Evaluation
from scipy.stats import pearsonr

# Text processing tools
from spellchecker import SpellChecker
from textblob import TextBlob

import language_check
from textstat.textstat import textstatistics, easy_word_set, legacy_round 


# For word embeddings and syntactic features
import spacy
import en_core_web_md

nlp = spacy.load('en_core_web_md')

# AutoML
from tpot import TPOTRegressor

#xgboost
from xgboost import XGBRegressor
import scipy
from scipy.stats.stats import pearsonr

#%% [markdown]
# ## Constants
# 
# Here we set some constant values related to local paths to data files as well as lists containing the various predictor and target features.

#%%
# Paths to various data targets for the competition. 

# Update to reflect the directory hierarchy of your machine 
DATA_DIR = "C:\\Users\\m0a00q3\\OneDrive - Walmart Inc\\SIOP 2019 - NLP Challenge\\Data"

TRAIN_CSV_DATA_NAME = "siop_ml_train_participant.csv"
TEST_CSV_DATA_NAME = "siop_ml_dev_participant.csv"
FINAL_CSV_DATA_NAME = "siop_ml_test_participant.csv"

# Set some DataFrame-specific constants
TARGET_COLUMN_NAMES = [attribute + "_Scale_score" for attribute in ["A", "E", "O", "N", "C"]]
PREDICTOR_TEXT_COLUMN_NAMES = ["open_ended_" + str(idx) for idx in range(1, 6)]
PREDICTOR_CONCAT_COLUMN_NAME = "open_ended_6"

#%% [markdown]
# ## Reading In Data
# 
# Here we use `pandas` to read our csv data sets into a DataFrame, a common and convenient data structure for the workflows we will be implementing. `df_train` will be used for training purposes, `df_test` will be used for public leaderboard submissions, and 'df_train' will be used for the private leaderboards submissions. 

#%%
# Read csv data to base DataFrame
df_train_temp = pd.read_csv(os.path.join(DATA_DIR, TRAIN_CSV_DATA_NAME))
df_test_temp = pd.read_csv(os.path.join(DATA_DIR, TEST_CSV_DATA_NAME))
df_final_temp = pd.read_csv(os.path.join(DATA_DIR, FINAL_CSV_DATA_NAME))

df_train_temp['Source']='Train'
df_test_temp['Source']='Test'
df_final_temp['Source']='Final'

# Combine datasets datasets
df_total=pd.concat([df_train_temp,df_test_temp,df_final_temp],ignore_index=True, sort=True)

# Check data load
df_total['Source'].value_counts()

#%% [markdown]
# ## Data Preprocessing Modules
# 
# Here we define various preprocessing utilities (simple python functions that operate on a single input) as well as preprocessing transformers which operate on an entire column of data. Transformers should be implemented as Python classes that inherit from `sklearn.base.BaseEstimator` and `sklearn.base.TransformerMixin` & should implement a `fit` and `transform` method.

#%%
df_total[PREDICTOR_CONCAT_COLUMN_NAME] = df_total.apply(
    lambda row: " ".join([row[col_name] for col_name in PREDICTOR_TEXT_COLUMN_NAMES]),
    axis=1
)

PREDICTOR_TEXT_COLUMN_NAMES_ALL =['open_ended_1','open_ended_2','open_ended_3',
                                  'open_ended_4','open_ended_5','open_ended_6']


#%%
# Count and Correct Spelling Errors. 

spell_checker = SpellChecker()

def tokenize(text):
    return TextBlob(text).words

def compute_num_spelling_errors(text):
    return len(spell_checker.unknown(tokenize(text)))

def divide(x, y):
    return x / y

def word_count(text): 
    return textstatistics().lexicon_count(text, removepunct=True)

for predictor_col in PREDICTOR_TEXT_COLUMN_NAMES_ALL:
    df_total[predictor_col + "_num_words"] = df_total[predictor_col].apply(word_count)
    df_total[predictor_col + "_num_misspelled"] = df_total[predictor_col].apply(compute_num_spelling_errors)
    df_total[predictor_col + "_percent_misspelled"] = df_total[[predictor_col + "_num_misspelled",
                              predictor_col + "_num_words"
    ]].apply(lambda x: divide(*x), axis=1)

#%% [markdown]
# ## Building word lists 1
# 
# We build the word lists twice because we were lazy. The lists diverged due to different team members refining them and we never got around to reconciling the differences.

#%%
# Lists were compliled by reading a sample of comments at either the top or bottom 5% of each trait distribution

O_high_5_LIST = ["accept","allow","apply","benefit","better","career","client","comfortable","contact","contribute","convince",
"correct","enjoy","excited","fair","first","fun","great time","grow","happy to go","help","immediately","improve","insist",
"leader","learn","let","mad","negative","no problem","offer","personal issue","respect","right away","show","team"]

C_high_2_LIST = ["family","report","stress","question","convince","job","deserve","longest","comfortable","win","great time",
"negative","fair","check in","short time","accus","short ","respect","willing","lie","correct","as soon","positive","impres",
"review","problems","immediately","hate networking","anger","proof","upset","prove","open","explain","improve","time ",
"confident","right away","let"]

A_high_1_LIST = ["agree","benefit","best","bonus","change","compromise","considerate","correct","defer","easy going","family",
"flexible","fun","good","help","hurt","incorrect","leader","let","misunderstand","no problem","not interested","obligation",
"paid","pick","priority","problems","quickly","respect","review","show","willing","win"]

E_high_3_LIST = ["career","good","frustrated","nice","best","deny","reflect","confident","grow","consequence","missed out",
"connect","rage","importan","worry","I am sociable","party","right away","priority","sociable","accept","focus","plan ",
"report","excited","reward","contribute","allow","success","contact","review","absolutely go","for sure go","meet","great",
"colleagues","social","not need anyone","regardless","fool","surely attend","leader","network","I like parties","no problem",
"learn","friendship","definitely go","introduce","let","competition","client", "make new friends"]

A_low_2_LIST = ["wrong","question","busy","probably","resent","not go","importan","fun","enjoy","bad","first","problems",
"refuse","better","short time","good","anxiety","avoid going","respect","compromise","losing","angry","regardless",
"social anxiety","rage","decline","pretend","focus","connect","no problem","priority","excuse","procrastinate","fool","sick",
"personal issue","anticipat","deadline","anyone","lose","difficult","meet","judg","worry","plan ","trouble","show","nervous",
"reflect","help","pressure","compensate","bonus","get along","flexible","colleagues","accus","fire","consequence","demand",
"not back down","stand my ground"]
            
A_low_3_LIST = ["I like parties","fire","unpleasant","would not go","sales","quit","discomfort","money","hate networking",
"worthwhile","obligation","panic","emotion","unlikely","hell","skip","social anxious","pick","cold","decline","not go","paid",
"get out of it","hate","wouldn't go","reward","rage","short ","negotiate","beg","difficult","trouble","resent","time ",
"immediately","stress","stressed","stressed out","reconsider","short time","grow","extremely uncomfortable","willing",
"get along","apply","if i had to","risk","anxiety","great","forc","allow","socially awkward","dislike","I am sociable",
"great time","missed out","compensate","oppurtunity","anger","benefit","plan ","confirm","avoid","social anxiety","fair",
"pressure","mad","deserve","not a social person"]
A_low_4_LIST = ["rage","cold","fool","marked","depend","demand","quit","report","probably","career","accept","not go",
"compensate","pressure","quiet","angry","afraid","confront","emotion","job","benefit","mad","threaten","money","unpleasant",
"anxiety","pissed","anyone","obligation","confident","short ","regardless","refuse","appeal","hesitate","examples","immediately",
"bad","suck it up","resent","respect","wrong","harm"]
A_low_5_LIST = ["paid","refuse","avoid going","alone","emotion","pretend","resent","bonus","win","rage","difficult","probably",
"afraid","anger","forc","hate networking","change","agree","depend","wouldn't go","pick","focus","obligation","frustrated",
"considerate","right away","time ","money","negative","colleagues","awkward","improve","success","explain","bad","best",
"respect","let","better","nice","nervous"]

N_low_1_LIST = ["as soon","report","show","problems","best","quickly","bonus","tense","social","correct","win","concede",
"leader","misunderstand","unlikely","incorrect","fire","easy going","paid","hesitate","human resources","time ","emotion",
"worried","racist","slash","fun","valid","stubborn","flexible","review","beg","respect","benefit","open","threaten","short ",
"change","first","trouble","agree","compromise","defend","defer","mad","harm","worry"]
N_low_2_LIST = ["hate networking","client","responsible","longest","unhappy","willing","accus","proof","difficult","family",
"anger","team","correct","consequence","comfortable","stick","trouble","job","pressure","benefit","mad","report","deserve",
"accept","positive","review","open","as soon","risk","time ","let","feel pressure","check in","depend","dislike","judg",
"social anxiety","resent","lie","explain","upset","hard ","leader","frustrated"]
N_low_3_LIST = ["learn","no problem","regardless","network","good","introduce","anyone","definitely go","confident","meet",
"competition","contact","lie","client","I like parties","not need anyone","great","social","party","worry","friendship",
"review","contribute","stretch myself","surely attend","fool","plan ","help","leader","missed out","for sure go","fair",
"let","reluctance","absolutely go","excited","happy to go","priority","excuse","hard ","report","job","anger"]
N_low_5_LIST = ["learn","anyone","short time","help","leader","client","great time","enjoy","importan","excited","hesitate",
"correct","lie","team","losing","career","responsible","insist","immediately","bad","happy to go","pretend","willing",
"emotion","short ","stress","confus","trouble","time ","worry","success","regardless","report","hurt","show","money",
"contact","stick","mad","unlikely"]

C_high_1_LIST = ["question","stubborn","would change","reconsider","as soon","human resources","disagree","defer","risk",
"unpleasant","immediately","worry","argue","petty","explain","mad","proof","hurt","correct","obligated","not go","harm",
"unhappy","leader","misunderstand","win","fire","unlikely","first","pick","angry","priority","bonus","quickly","short time",
"hesitate","tense","social","switch","success","problems","not interested","easy going","no problem","report","reflect",
"upset","anger","team","valid","paid","review","agree","short ","willing","fun","concede","show","seniority","flexible",
"change"]
C_high_3_LIST = ["no problem","introvert","frustrated","win","sad","missed out","alone","dislike","appeal","depend","insist",
"sick","success","not comfortable","stretch myself","report","benefit","accept","hard ","contribute","responsible","compensate",
"fool","social","absolutely go","regardless","anyone","focus","pretend","worried","nightmare","not need anyone","surely attend",
"for sure go","colleagues","competition","let","help","best","great","deny","importan","learn","network","client","uncomfortable",
"priority","lie","improve","good","not attend","fun","definitely go","mad","comfortable","reluctant","excited","excuse","meet"]
C_high_4_LIST = ["willing","change","respect","connect","fun","paid","hard ","immediately","terrible","grow","incorrect","refuse",
"open","resent","quickly","contact","calm","party","short ","contribute","my right","stubborn","rebut","problems","worried",
"as soon","compromise","hurt","good","proof","not true","early","human resources""obligation","colleagues","meet","demand",
"success","negative","allow","concerned","disagree","let","agree"]
C_high_5_LIST = ["success","appeal","worry","fun","busy","hesitate","problems","allow","hurt","improve","excited","good","bad",
"leader","stress","importan","excuse","introduce","lose","enjoy","prove","personal issue","fair","quickly","correct","stick",
"accus","unlikely","comfortable","sad","willing","contact","confus","career","show","losing","immediately","compensate",
"anyone","lie","client","help","learn"]

A_high_2_LIST = ["agree","negative","benefit","overwhelmed","quiet","I had to","lie","team","check in","early","stick",
"feel pressure","allow","family","sacrifice","stressed","learn","frustrated","right away","convince","best","let","fair",
"client","longest","responsible","mad","stressed out","report","time ","upset","confident","dislike","unhappy","anger",
"explain","positive","stress","proof","avoid confrontation","more than willing","don't want conflict","easy going",
"hate conflict","keep people happy","team player"]
A_high_3_LIST = ["change","reluctant","angry","quickly","right away","excuse","stick","would change","early","compromise",
"not comfortable","learn","positive","avoid going","anxious","colleagues","fool","reluctance","absolutely go","fun",
"not attend","tired","losing","worry","busy","no problem","contribute","explain","hurt","network","uncomfortable","consequence",
"social","not need anyone","surely attend","regardless","help","better","excited","importan","priority","responsible",
"outside of my comfort zone","party","stretch myself","for sure go","hard ","report","focus","client","alone","lie","introduce",
"friendship","comfortable","contact","best","good","definitely go","anyone","meet"]
A_high_4_LIST = ["convince","defend","lose","accus","worthwhile","agitated","personal","consequence","concerned","impres",
"anger","success","correct","win","confus","argue","proof","incorrect","focus","terrible","best","negative","not justified",
"as soon","plead","confirm","lie","unfair","early","judg","stressed out","hard ","organize","risk","improve","worried","quickly",
"my right","open","frustrated","contact","meet","compromise","pretend","rebut","stress","reconsider","hurt","would not go",
"importan","positive","problems","agree","let","negotiate","allow","explain","learn","prove","better","anxious","colleagues",
"not true","upset","grow"]
A_high_5_LIST = ["accept","short time","question","happy to go","excited","hard ","impres","good","grow","losing","reward",
"show","contribute","convince","accus","willing","concerned","dislike","contact","hesitate","network","comfortable","apply",
"leader","immediately","stress","correct","importan","great time","hurt","offer","confus","help","anyone","lie","client",
"enjoy","learn"]

N_high_1_LIST = ["willing","regardless","losing","great","shy","career","obligated","organize","stick","forc","appeal","anger",
"unfair","positive","early","reward","my right","I had to","refuse","money","negotiate","personal issue","wrong","anyone",
"family","enjoy","pissed","hard ","team","deny","insist","busy","sacrifice","skip","proof","fair","client","better","contact",
"meet","question","fool","get even","profanity","cold","unhappy","angry","call in","awkward","excuse","upset","get along",
"demand","lose","avoid","deadline","stressed","unpleasant","terrible","difficult","frustrated","confront","hell","plead",
"alone","improve","stare","concerned","hardship","nice","pressure","sad","reflect","probably","friendship","reluctant",
"sick","obligation","quit","hate","offer","hard stance"]
N_high_2_LIST = ["calm","panic","stress","enjoy","bonus","show","learn","question","decline","sick","importan","colleagues",
"worried","worry","connect","meet","rage","paid","pretend","anxiety","avoid going","lose","early","bad","angry","better",
"deadline","losing","hurt","priority","no problem","wrong","demand","beg","I had to","busy","compromise","negotiate","probably"]
N_high_3_LIST = ["wrong","compromise","respect","risk","show","afraid","bonus","worried","tense","worthwhile","dislike","valid",
"confirm","socially awkward","introvert","losing","deserve","quickly","beg","plead","mad","change","better","angry","apply",
"not comfortable","lose","get out of it","agree","paid","outside of my comfort zone","not a social person","miss out",
"time ","family","reconsider","anxious","short time","prove","negative","money","fire","tired","negotiate","I had to","harm",
"appeal","sacrifice","hell","stress","awkward","forc","hesitate","pressure","trouble","willing","deadline","short ","suck it up",
"get along","loner","stressed","resent","skip","social anxiety","bad","not great at networking","nightmare","shy","avoid",
"impres","concerned","difficult","probably","compensate","emotion","unpleasant","obligation","nervous","feel pressure",
"extremely uncomfortable","nerve-wracking","hate networking","immediately","hate","would not go","social anxious","panic",
"unlikely","discomfort","not go","anxiety"]
N_high_5_LIST = ["negative","allow","best","hate networking","let","positive","apply","anger","beg","bonus","comfortable",
"dislike","oppurtunity","obligation","improve","concerned","pick","open","right away","job","rage","probably","refuse",
"upset","afraid","risk","alone","social anxiety","consequence","agree","prove","fair","colleagues","awkward","paid","grow",
"avoid going","early","nervous","forc","depend","resent","frustrated","difficult"]

O_high_1_LIST = ["accept","anyone","as soon","best","bonus","defer","easy going","enjoy","excuse","flexible","fool","get even",
"good","harm","hesitate","hurt","importan","leader","marked","meet","negotiate","obligation","paid","petty","pick","plead",
"positive","probably","problems","quickly","quit","reflect","respect","reward","short ","short time","stubborn","suck it up",
"suffer","tense","terrible","threaten","time ","upset","willing","win","worried"]
O_high_2_LIST = ["agree","allow","anger","best","better","calm","correct","deserve","difficult","excuse","explain","fair",
"forc","frustrated","fun","great time","immediately","importan","improve","learn","let","nervous","offer","pick","positive",
"pressure","problems","proof","prove","respect","responsible","review","short ","short time","show","suffer","team","time ",
"trouble","upset","worried"]
O_high_3_LIST = ["absolutely go","accept","alone","angry","anyone","better","career","client","cold","comfortable","consequence",
"contact","contribute","definitely go","deny","depend","difficult","early","emotion","excited","excuse","feel pressure","focus",
"for sure go","forc","friendship","fun","good","hesitate","I like parties","insist","introduce","let","lie","lose","losing",
"meet","miss out","missed out","money","nerve-wracking","nervous","network","nice","no problem","not comfortable","not need anyone",
"obligated","oppurtunity","outside of my comfort zone","party","plan ","positive","priority","quickly","regardless","responsible",
"success","trouble","worried","worry","would change"]
O_high_4_LIST = ["allow","anxious","as soon","benefit","best","better","bonus","client","cold","colleagues","comfortable",
"concerned","confirm","connect","consequence","deserve","early","explain","fool","forc","fun","great time","grow","help",
"importan","impres","improve","judg","learn","let","lie","losing","marked","meet","negotiate","nervous","nice","not justified",
"not true","offer","paid","party","personal","personal issue","plan ","positive","pretend","problems","prove","quiet",
"reconsider","resent","respect","review","risk","stick","stubborn","team","threaten","trouble"]

A_low_1_LIST = ["stare","responsible","fool","get even","profanity","call in","sick","refuse","emotion","hard stance","racist",
"slash","hardship","demand","compensate","first","stick","quit","personal issue","excuse","trouble","deny","hell","depend",
"money","cold","hard ","marked","pissed","client","deserve","unfair","fair","resent","reconsider","offer","my right","hate",
"forc","worry","reward","reluctant","concerned","organize","sad","losing","rage","bad","insist","busy","difficult","appeal",
"stressed out","stressed","wrong","early","longest","proof","better","petty","improve","contact","avoid","accept","entitle",
"meet","if i had to","seniority","suffer","comfortable","regardless","personal"]

E_low_3_LIST = ["social anxiety","extremely uncomfortable","nervous","social anxious","panic","unlikely","impres","anxiety",
"probably","introvert","immediately","feel pressure","anxious","decline","emotion","nerve-wracking","loner","pressure","avoid",
"stressed out","stressed","dislike","shy","hesitate","losing","bad","difficult","not great at networking","obligation","unfair",
"stretch myself","hate networking","willing","hell","stress","lose","nightmare","quit","avoid going","mad","paid","sad",
"reluctant","get out of it","fair","not a social person","reluctance","quiet","upset","sacrifice","change","not comfortable",
"money","tired","family","appeal","confirm","harm","prove","short ","skip","stick","hate","compensate","deserve","short time",
"sick","outside of my comfort zone","deadline","pretend","discomfort","socially awkward","show","angry","win","convince",
"not interested","apply","get along","negotiate","unpleasant","quickly","awkward","not attend","concerned","plead","fire",
"suck it up","forc","comfortable","pick","uncomfortable","unhappy","excuse","compromise","afraid","do not interact well with strangers",
"don't like being in social situations","don't like networking","don't like socializing","very shy"]

GO_3_LIST = ["absolutely go","all in","attend","attend that meeting","certainly go","cheerfully go","decide to go",
"definitely attend","definitely be in attendance_1","definitely go","definitely still go","go for it","go for sure",
"go to the event","go to the meeting","go to the networking meeting","just go","make an appearance","make sure I go",
"make time to attend","still attend","still go","still opt in","time and go","would attend","would go","would still go"]

NOGO_3_LIST = ["avoid","backing out","bow out of the meeting","choose not to go","come","consider not going","decide to go",
"decline","ditch","get out of it","go home","happy to go","hate going","hesitate to go","in attendance","likely go",
"likely not go","might consider going","no interest","not attend","not come","not consider going","not feel like going",
"not going","not interested","not show up","not volunteer","not want to go","politely decline","probably attend","probably go",
"probably not go","probably still go","probably would not","probably wouldn't","skip","stay at home","try to go","unlikely to go",
"will not go","would be going","would not go","wouldn't be going","wouldn't do it","wouldn't go","wouldn't want to go"]

GO_5_LIST = ['would go','probably go']

NOGO_5_LIST = ['not go','not to go',"n't go"]

NOT_LIST = [" not "]

NO_LIST = [" no "]


#%%
# Define function for counting word occurance

def write_keyword_count_column(df, target_column, source_column, keyword_list):
    def compute_keyword_list_count(text):
        return sum([text.count(kw) for kw in keyword_list])    
    df[target_column] = df[source_column].apply(compute_keyword_list_count)


#%%
# Specify key word list features

write_keyword_count_column(df_total, 'O_high_5', 'open_ended_5', O_high_5_LIST)

write_keyword_count_column(df_total, 'C_high_2', 'open_ended_2', C_high_2_LIST)

write_keyword_count_column(df_total, 'A_high_1', 'open_ended_1', A_high_1_LIST)

write_keyword_count_column(df_total, 'E_high_3', 'open_ended_3', E_high_3_LIST)

write_keyword_count_column(df_total, 'A_low_2', 'open_ended_2', A_low_2_LIST)
write_keyword_count_column(df_total, 'A_low_3', 'open_ended_3', A_low_3_LIST)
write_keyword_count_column(df_total, 'A_low_4', 'open_ended_4', A_low_4_LIST)
write_keyword_count_column(df_total, 'A_low_5', 'open_ended_5', A_low_5_LIST)

write_keyword_count_column(df_total, 'N_low_1', 'open_ended_1', N_low_1_LIST)
write_keyword_count_column(df_total, 'N_low_2', 'open_ended_2', N_low_2_LIST)
write_keyword_count_column(df_total, 'N_low_3', 'open_ended_3', N_low_3_LIST)
write_keyword_count_column(df_total, 'N_low_5', 'open_ended_5', N_low_5_LIST)

write_keyword_count_column(df_total, 'C_high_1', 'open_ended_1', C_high_1_LIST)
write_keyword_count_column(df_total, 'C_high_3', 'open_ended_3', C_high_3_LIST)
write_keyword_count_column(df_total, 'C_high_4', 'open_ended_4', C_high_4_LIST)
write_keyword_count_column(df_total, 'C_high_5', 'open_ended_5', C_high_5_LIST)

write_keyword_count_column(df_total, 'A_high_2', 'open_ended_1', A_high_2_LIST)
write_keyword_count_column(df_total, 'A_high_3', 'open_ended_3', A_high_3_LIST)
write_keyword_count_column(df_total, 'A_high_4', 'open_ended_4', A_high_4_LIST)
write_keyword_count_column(df_total, 'A_high_5', 'open_ended_5', A_high_5_LIST)

write_keyword_count_column(df_total, 'N_high_1', 'open_ended_1', N_high_1_LIST)
write_keyword_count_column(df_total, 'N_high_2', 'open_ended_2', N_high_2_LIST)
write_keyword_count_column(df_total, 'N_high_3', 'open_ended_3', N_high_3_LIST)
write_keyword_count_column(df_total, 'N_high_5', 'open_ended_5', N_high_5_LIST)

write_keyword_count_column(df_total, 'O_high_1', 'open_ended_1', O_high_1_LIST)
write_keyword_count_column(df_total, 'O_high_2', 'open_ended_2', O_high_2_LIST)
write_keyword_count_column(df_total, 'O_high_3', 'open_ended_3', O_high_3_LIST)
write_keyword_count_column(df_total, 'O_high_4', 'open_ended_4', O_high_4_LIST)

write_keyword_count_column(df_total, 'E_high_3', 'open_ended_3', O_high_2_LIST)
write_keyword_count_column(df_total, 'E_high_4', 'open_ended_4', O_high_3_LIST)
write_keyword_count_column(df_total, 'E_high_5', 'open_ended_5', O_high_4_LIST)

write_keyword_count_column(df_total, 'A_low_1', 'open_ended_1', A_low_1_LIST)

write_keyword_count_column(df_total, 'E_low_3', 'open_ended_3', E_low_3_LIST)

write_keyword_count_column(df_total, 'GO_3', 'open_ended_3', GO_3_LIST)
write_keyword_count_column(df_total, 'NOGO_3', 'open_ended_3', NOGO_3_LIST)

write_keyword_count_column(df_total, 'GO_5', 'open_ended_5', GO_3_LIST)
write_keyword_count_column(df_total, 'NOGO_5', 'open_ended_5', NOGO_3_LIST)

write_keyword_count_column(df_total, 'NOT_1', 'open_ended_1', NOT_LIST)
write_keyword_count_column(df_total, 'NOT_2', 'open_ended_2', NOT_LIST)
write_keyword_count_column(df_total, 'NOT_3', 'open_ended_3', NOT_LIST)
write_keyword_count_column(df_total, 'NOT_4', 'open_ended_4', NOT_LIST)
write_keyword_count_column(df_total, 'NOT_5', 'open_ended_5', NOT_LIST)

write_keyword_count_column(df_total, 'NO_5', 'open_ended_5', NO_LIST)
write_keyword_count_column(df_total, 'NOT_5', 'open_ended_5', NOT_LIST)


#%%
# Generating aggregate features--combinations were derived in part from feedback from the public leaderboard

df_total['A_low_comb'] = df_total['A_low_2']+df_total['A_low_3']+df_total['A_low_4']+df_total['A_low_5']
df_total['N_low_comb'] = df_total['N_low_1']+df_total['N_low_2']+df_total['N_low_3']+df_total['N_low_5']
df_total['C_high_comb'] = df_total['C_high_1']+df_total['C_high_3']+df_total['C_high_4']+df_total['C_high_5']
df_total['A_high_comb'] = df_total['A_high_2']+df_total['A_high_3']+df_total['A_high_4']+df_total['A_high_5']
df_total['N_high_comb'] = df_total['N_high_1']+df_total['N_high_2']+df_total['N_high_3']+df_total['N_high_5']
df_total['O_high_comb'] = df_total['O_high_1']+df_total['O_high_2']+df_total['O_high_3']+df_total['O_high_4']
df_total['E_high_3to5'] = df_total['E_high_3']+df_total['E_high_4']+df_total['E_high_5']

df_total['A_not_comb'] = df_total['NOT_1']+df_total['NOT_2']+df_total['NOT_3']+df_total['NOT_4']+df_total['NOT_5']

df_total['O_go_comb'] = df_total['GO_5']-df_total['NOGO_5']

#%% [markdown]
# ## Building word lists 2

#%%
df_total['char_count_3'] = df_total['open_ended_3'].str.len() 
df_total['char_count_4'] = df_total['open_ended_4'].str.len()


#%%
def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

df_total['avg_word_1'] = df_total['open_ended_1'].apply(lambda x: avg_word(x))
df_total['avg_word_2'] = df_total['open_ended_2'].apply(lambda x: avg_word(x))
df_total['avg_word_3'] = df_total['open_ended_3'].apply(lambda x: avg_word(x))
df_total['avg_word_4'] = df_total['open_ended_4'].apply(lambda x: avg_word(x))
df_total['avg_word_5'] = df_total['open_ended_5'].apply(lambda x: avg_word(x))


#%%
not_list = [" not "]

no_list = [" no "] #apply to 5 only

e_high_3_list=['benefit','best','better','career','client','competition','confident','connect',
'contact','contribute','convince','drink','enjoy','friendship','good','great','grow','importan',
'impres','introduce','leader','learn','meet','miss out','missed out','network','open','oppurtunity',
'party','positive','regardless','reward','sales','show','sociable','success','worthwhile']

e_high_4_list=['benefit','best','better','career','client','competition','confident','connect',
'contact','contribute','convince','drink','enjoy','friendship','good','great','grow','importan',
'impres','introduce','leader','learn','meet','miss out','missed out','network','open','oppurtunity',
'party','positive','regardless','reward','sales','show','sociable','success','worthwhile']

e_high_5_list =['benefit','best','better','career','client','competition','confident','connect',
'contact','contribute','convince','drink','enjoy','friendship','good','great','grow','importan',
'impres','introduce','leader','learn','meet','miss out','missed out','network','open','oppurtunity',
'party','positive','regardless','reward','sales','show','sociable','success','worthwhile']

n_low_comb_emp_1_list=['as soon','report','show','problems','best','quickly','bonus','tense',
'social','correct','win','concede','leader','misunderstand','unlikely','incorrect','fire',
'easy going','paid','hesitate','human resources','time ','emotion','worried','racist','slash',
'fun','valid','stubborn','flexible','review','beg','respect','benefit','open','threaten','short ',
'change','first','trouble','agree','compromise','defend','defer','mad','harm']

n_low_comb_emp_2_list=['worry','hate networking','client','responsible','longest','unhappy',
'willing','accus','proof','difficult','family','anger','team','correct','consequence','comfortable',
'stick','trouble','job','pressure','benefit','mad','report','deserve','accept','positive','review',
'open','as soon','risk','time ','let','feel pressure','check in','depend','dislike','judg','social anxiety',
'resent','lie','explain','upset','hard ','leader']

n_low_comb_emp_3_list=['frustrated','learn','no problem','regardless','network','good','introduce',
'anyone','definitely go','confident','meet','competition','contact','lie','client','I like parties',
'not need anyone','great','social','party','worry','friendship','review','contribute','stretch myself',
'surely attend','fool','plan ','help','leader','missed out','for sure go','fair','let','reluctance',
'absolutely go','excited','happy to go','priority','excuse','hard ','report','job']

n_low_comb_emp_5_list=['anger','learn','anyone','short time','help','leader','client','great time',
'enjoy','importan','excited','hesitate','correct','lie','team','losing','career','responsible',
'insist','immediately','bad','happy to go','pretend','willing','emotion','short ','stress','confus',
'trouble','time ','worry','success','regardless','report','hurt','show','money','contact','stick',
'mad','unlikely']

n_high_comb_emp_1_list=['willing','regardless','losing','great','shy','career','obligated','organize',
'stick','forc','appeal','anger','unfair','positive','early','reward','my right','I had to','refuse',
'money','negotiate','personal issue','wrong','anyone','family','enjoy','pissed','hard ','team',
'deny','insist','busy','sacrifice','skip','proof','fair','client','better','contact','meet','question',
'fool','get even','profanity','cold','unhappy','angry','call in','awkward','excuse','upset','get along',
'demand','lose','avoid','deadline','stressed','unpleasant','terrible','difficult','frustrated','confront',
'hell','plead','alone','improve','stare','concerned','hardship','nice','pressure','sad','reflect','probably',
'friendship','reluctant','sick','obligation','quit','hate','offer','hard stance']

n_high_comb_emp_2_list=['calm','panic','stress','enjoy','bonus','show','learn','question','decline',
'sick','importan','colleagues','worried','worry','connect','meet','rage','paid','pretend','anxiety',
'avoid going','lose','early','bad','angry','better','deadline','losing','hurt','priority','no problem',
'wrong','demand','beg','I had to','busy','compromise','negotiate','probably']

n_high_comb_emp_3_list=['wrong','compromise','respect','risk','show','afraid','bonus','worried','tense',
'worthwhile','dislike','valid','confirm','socially awkward','introvert','losing','deserve','quickly',
'beg','plead','mad','change','better','angry','apply','not comfortable','lose','get out of it',
'agree','paid','outside of my comfort zone','not a social person','miss out','time ','family','reconsider',
'anxious','short time','prove','negative','money','fire','tired','negotiate','I had to','harm','appeal',
'sacrifice','hell','stress','awkward','forc','hesitate','pressure','trouble','willing','deadline','short ',
'suck it up','get along','loner','stressed','resent','skip','social anxiety','bad','not great at networking',
'nightmare','shy','avoid','impres','concerned','difficult','probably','compensate','emotion','unpleasant',
'obligation','nervous','feel pressure','extremely uncomfortable','nerve-wracking','hate networking','immediately',
'hate','would not go','social anxious','panic','unlikely','discomfort','not go','anxiety']

n_high_comb_emp_5_list=['negative','allow','best','hate networking','let','positive','apply','anger',
'beg','bonus','comfortable','dislike','oppurtunity','obligation','improve','concerned','pick','open',
'right away','job','rage','probably','refuse','upset','afraid','risk','alone','social anxiety','consequence',
'agree','prove','fair','colleagues','awkward','paid','grow','avoid going','early','nervous','forc',
'depend','resent','frustrated','difficult']

e_low_3_emp_list=['social anxiety','extremely uncomfortable','nervous','social anxious','panic','unlikely',
'impres','anxiety','probably','introvert','immediately','feel pressure','anxious','decline','emotion',
'nerve-wracking','loner','pressure','avoid','stressed out','stressed','dislike','shy','hesitate','losing',
'bad','difficult','not great at networking','obligation','unfair','stretch myself','hate networking',
'willing','hell','stress','lose','nightmare','quit','avoid going','mad','paid','sad','reluctant','get out of it',
'fair','not a social person','reluctance','quiet','upset','sacrifice','change','not comfortable','money',
'tired','family','appeal','confirm','harm','prove','short ','skip','stick','hate','compensate','deserve',
'short time','sick','outside of my comfort zone','deadline','pretend','discomfort','socially awkward',
'show','angry','win','convince','not interested','apply','get along','negotiate','unpleasant','quickly',
'awkward','not attend','concerned','plead','fire','suck it up','forc','comfortale','pick','uncomfortable',
'unhappy','excuse','compromise','afraid','do not interact well with strangers',"don't like being in social situation",
"don't like networking","don't like socializing",'very shy']    

e_high_3_emp_list=['career','good','frustrated','nice','best','deny','reflect','confident','grow','consequence',
'missed out','connect','rage','importan','worry','I am sociable','party','right away','priority','sociable',
'accept','focus','plan ','report','excited','reward','contribute','allow','success','contact','review',
'absolutely go','for sure go','meet','great','colleagues','social','not need anyone','regardless','fool',
'surely attend','leader','network','I like parties','no problem','learn','friendship','definitely go','introduce',
'let','competition','client','make new friends']

c_high_2_emp_list=['family','report','stress','question','convince','job','deserve','longest','comfortable','win','great time',
'negative','fair','check in','short time','accus','short ','respect','willing','lie','correct','as soon',
'positive','impres','review','problems','immediately','hate networking','anger','proof','upset','prove',
'open','explain','improve','time ','confident','right away','let']    

c_low_comb_emp_1_list=['hard stance','hardship','my right','call in','stare','contact','shy','quit','entitle','sad','reluctant','refuse',
'demand','fool','wrong','get even','profanity','compensate','pressure','convince','responsible','fair','prove',
'client','skip','get along','negotiate','sick','family','hell','difficult','sacrifice','obligation','awkward',
'offer','friendship','hard ','career','threaten','party','allow','avoid','stick','hate','improve','terrible',
'deadline','plan ','great','personal','plead','trouble','marked','respect','probably','early','confront',
'lose','suck it up','better','excuse','suffer','busy','money','stress','beg','unfair','time ','personal issue',
'deny','stressed']
    
c_low_comb_emp_3_list=['obligation','unlikely','panic','would not go','discomfort','deserve','awkward','immediately','unhappy',
'social anxiety','skip','forc','social anxious','harm','probably','unpleasant','worthwhile','oppurtunity',
'fire','negative','get along','shy','team','short ','open','pick','feel pressure','show','quit','angry',
'avoid','hell','confirm','decline','money','socially awkward','negotiate','extremely uncomfortable','time ',
'short time','not go','better','avoid going','not a social person','loner','valid','anger','hesitate',
'hate networking','great time','positive','sociable','prove','emotion','compromise','allow','rage','concerned',
'anxiety','outside of my comfort zone','wrong','deadline','stress','resent','personal issue','reconsider',
'cold','not interested','unfair','early','drink','stressed']

c_low_comb_emp_4_list=['probably','career','appeal','anxiety','depend','wrong','sad','cold','fool','marked','reflect','not go',
'harm','bad','hate networking','job','mad','money','would change','reward','quit','focus','organize',
'stressed','hesitate','leader','valid','difficult','consequence','emotion','personal issue','bonus',
'accept','review','nice','avoid going','plan ','great time','lose','fire','frustrated','pressure','confront']
    
c_low_comb_emp_5_list=['forc','network','resent','considerate','difficult','I had to','frustrated','paid','obligation','nervous',
'refuse','explain','rage','anger','grow','meet','respect','oppurtunity','bonus','nice','insist','job',
'depend','avoid going','upset','awkward','positive','apply','short ','connect','probably','plan ','win',
'open','concerned','question','negative','change','friendship','dislike','focus','alone']
    
c_high_comb_emp_1_list=['question','stubborn','would change','reconsider','as soon','human resources','disagree','defer','risk',
'unpleasant','immediately','worry','argue','petty','explain','mad','proof','hurt','correct','obligated',
'not go','harm','unhappy','leader','misunderstand','win','fire','unlikely','first','pick','angry','priority',
'bonus','quickly','short time','hesitate','tense','social','switch','success','problems','not interested',
'easy going','no problem','report','reflect','upset','anger','team','valid','paid','review','agree',
'short ','willing','fun','concede','show','seniority','flexible','change']
    
c_high_comb_emp_3_list=['no problem','introvert','frustrated','win','sad','missed out','alone','dislike','appeal','depend','insist',
'sick','success','not comfortable','stretch myself','report','benefit','accept','hard ','contribute','responsible',
'compensate','fool','social','absolutely go','regardless','anyone','focus','pretend','worried','nightmare',
'not need anyone','surely attend','for sure go','colleagues','competition','let','help','best','great',
'deny','importan','learn','network','client','uncomfortable','priority','lie','improve','good','not attend',
'fun','definitely go','mad','comfortable','reluctant','excited','excuse','meet']
    
c_high_comb_emp_4_list=['willing','change','respect','connect','fun','paid','hard ','immediately','terrible','grow','incorrect',
'refuse','open','resent','quickly','contact','calm','party','short ','contribute','my right','stubborn',
'rebut','problems','worried','as soon','compromise','hurt','good','proof','not true','early','human resources',
'obligation','colleagues','meet','demand','success','negative','allow','concerned','disagree','let','agree']    
    

c_high_comb_emp_5_list=['success','appeal','worry','fun','busy','hesitate','problems','allow','hurt','improve','excited','good',
'bad','leader','stress','importan','excuse','introduce','lose','enjoy','prove','personal issue','fair',
'quickly','correct','stick','accus','unlikely','comfortable','sad','willing','contact','confus','career',
'show','losing','immediately','compensate','anyone','lie','client','help','learn']
    
a_low_emp_1_list=['stare','responsible','fool','get even','profanity','call in','sick','refuse','emotion','hard stance','racist',
'slash','hardship','demand','compensate','first','stick','quit','personal issue','excuse','trouble','deny',
'hell','depend','money','cold','hard ','marked','pissed','client','deserve','unfair','fair',
'resent','reconsider','offer','my right','hate','forc','worry','reward','reluctant','concerned','organize',
'sad','losing','rage','bad','insist','busy','difficult','appeal','stressed out','stressed','wrong',
'early','longest','proof','better','petty','improve','contact','avoid','accept','entitle','meet',
'if I had to','seniority','suffer','comfortable','regardless','personal','not back down','stand my ground']  
    
a_high_emp_1_list=['agree','benefit','best','bonus','change','compromise','considerate','correct','defer','easy going','family',
'flexible','fun','good','help','hurt','incorrect','leader','let','misunderstand','no problem','not interested',
'obligation','paid','pick','priority','problems','quickly','respect','review','show','willing','win',
'more than willing',"don't want conflict",'easy going','hate conflict','team player','avoid confrontation',
'keep people happy']  
    
a_low_comb_emp_2_list=['wrong','question','busy','probably','resent','not go','importan','fun','enjoy','bad','first','problems',
'refuse','better','short time','good','anxiety','avoid going','respect','compromise','losing','angry',
'regardless','social anxiety','rage','decline','pretend','focus','connect','no problem','priority','excuse',
'procrastinate','fool','sick','personal issue','anticipat','deadline','anyone','lose','difficult','meet',
'judg','worry','plan ','trouble','show','nervous','reflect','help','pressure','compensate','bonus','get along',
'flexible','colleagues','accus','fire','consequence','demand']
    
a_low_comb_emp_3_list=['I like parties','fire','unpleasant','would not go','sales','quit','discomfort','money','hate networking',
'worthwhile','obligation','panic','emotion','unlikely','hell','skip','social anxious','pick','cold',
'decline','not go','paid','get out of it','hate','reward','rage','short ','negotiate','beg','difficult',
'trouble','resent','time ','immediately','stress','stressed','stressed out','reconsider','short time',
'grow','extremely uncomfortable','willing','get along','apply','I had to','risk','anxiety','great',
'forc','allow','socially awkward','dislike','I am sociable','great time','missed out','compensate','oppurtunity',
'anger','benefit','plan ','confirm','avoid','social anxiety','fair','pressure','mad','deserve',
'not a social person']
    
a_low_comb_emp_4_list=['rage','cold','fool','marked','depend','demand','quit','report','probably','career','accept','not go',
'compensate','pressure','quiet','angry','afraid','confront','emotion','job','benefit','mad','threaten',
'money','unpleasant','anxiety','pissed','anyone','obligation','confident','short ','regardless','refuse','appeal',
'hesitate','examples','immediately','bad','suck it up','resent','respect','wrong','harm']
    
a_low_comb_emp_5_list=['paid','refuse','avoid going','alone','emotion','pretend','resent','bonus','win','rage','difficult',
'probably','afraid','anger','forc','hate networking','change','agree','depend','pick','focus','obligation',
'frustrated','considerate','right away','time ','money','negative','colleagues','awkward','improve','success',
'explain','bad','best','respect','let','better','nice','nervous']
    
a_high_comb_emp_2_list=['agree','negative','benefit','overwhelmed','quiet','I had to','lie','team','check in','early','stick',
'feel pressure','allow','family','sacrifice','stressed','learn','frustrated','right away','convince',
'best','let','fair','client','longest','responsible','mad','stressed out','report','time ','upset',
'confident','dislike','unhappy','anger','explain','positive','stress','proof']
    
a_high_comb_emp_3_list=['change','reluctant','angry','quickly','right away','excuse','stick','would change','early','compromise',
'not comfortable','learn','positive','avoid going','anxious','colleagues','fool','reluctance','absolutely go',
'fun','not attend','tired','losing','worry','busy','no problem','contribute','explain','hurt','network',
'uncomfortable','consequence','social','not need anyone','surely attend','regardless','help','better',
'excited','importan','priority','responsible','outside of my comfort zone','party','stretch myself','for sure go',
'hard ','report','focus','client','alone','lie','introduce','friendship','comfortable','contact',
'best','good','definitely go','anyone','meet']  
    
a_high_comb_emp_4_list=['convince','defend','lose','accus','worthwhile','agitated','personal','consequence','concerned','impres',
'anger','success','correct','win','confus','argue','proof','incorrect','focus','terrible','best',
'negative','not justified','as soon','plead','confirm','lie','unfair','early','judg','stressed out','hard ',
'organize','risk','improve','worried','quickly','my right','open','frustrated','contact','meet',
'compromise','pretend','rebut','stress','reconsider','hurt','would not go','importan','positive','problems',
'agree','let','negotiate','allow','explain','learn','prove','better','anxious','colleagues','not true',
'upset','grow']
       
a_high_comb_emp_5_list=['accept','short time','question','happy to go','excited','hard ','impres','good','grow','losing',
'reward','show','contribute','convince','accus','willing','concerned','dislike','contact','hesitate',
'network','comfortable','apply','leader','immediately','stress','correct','importan','great time','hurt',
'offer','confus','help','anyone','lie','client','enjoy','learn']

go_v2_3_list=['go for it','make an appearance','certainly go','would attend','just go','attend','still attend','still go',
'would still go','definitely go','would go','all in','definitely be in attendance','absolutely go',
'attend that meeting','cheerfully go','decide to go','definitely attend','definitely still go','go for sure',
'go to the event','go to the meeting','go to the networking meeting','make sure I go','make time to attend',
'still opt in','time and go']
    
not_go_v2_3_list=['would not go',"wouldn't go",'probably not go',"wouldn't want to go",'unlikely to go','decline',
'not show up','hesitate to go','go home','ditch','avoid','get out of it','probably still go',
'probably go','skip','decide to go','try to go','not going','in attendance','not interested',
'come','not attend','would be going','not come','likely go','happy to go','probably attend',
'bow out of the meeting','choose not to go','consider not going','hate going','likely not go',
'not consider going','not feel like going','not want to go','politely decline','probably would not',
"probably wouldn't",'stay at home','will not go']
    
not_go_5_list=['not go','not to go',"n't go"]

go_5_list=['would go','probably go']

#%% [markdown]
# ## Word List Section
# 
# This section includes the code that was used in the word list prediction. The optimal weights were derived from feedback on the public leaderboard.

#%%
# Compute new word count variables from word lists

write_keyword_count_column(df_total, 'not_go_5', 'open_ended_5', not_go_5_list)
write_keyword_count_column(df_total, 'go_5', 'open_ended_5', go_5_list)

df_total['go_comb_5']=df_total['go_5']-df_total['not_go_5']

write_keyword_count_column(df_total, 'not_1', 'open_ended_1', not_list)
write_keyword_count_column(df_total, 'not_2', 'open_ended_2', not_list)
write_keyword_count_column(df_total, 'not_3', 'open_ended_3', not_list)
write_keyword_count_column(df_total, 'not_4', 'open_ended_4', not_list)
write_keyword_count_column(df_total, 'not_5', 'open_ended_5', not_list)

df_total['sum_not']=df_total['not_1']+df_total['not_2']+df_total['not_3']+df_total['not_4']+df_total['not_5']

write_keyword_count_column(df_total, 'no_5', 'open_ended_5', no_list)

write_keyword_count_column(df_total, 'n_low_comb_emp_1', 'open_ended_1', n_low_comb_emp_1_list)
write_keyword_count_column(df_total, 'n_low_comb_emp_2', 'open_ended_2', n_low_comb_emp_2_list)
write_keyword_count_column(df_total, 'n_low_comb_emp_3', 'open_ended_3', n_low_comb_emp_3_list)
write_keyword_count_column(df_total, 'n_low_comb_emp_5', 'open_ended_5', n_low_comb_emp_5_list)

df_total['n_low_comb_emp']=df_total['n_low_comb_emp_1']+df_total['n_low_comb_emp_2']+df_total['n_low_comb_emp_3']+df_total['n_low_comb_emp_5']

write_keyword_count_column(df_total, 'n_high_comb_emp_1', 'open_ended_1', n_high_comb_emp_1_list)
write_keyword_count_column(df_total, 'n_high_comb_emp_2', 'open_ended_2', n_high_comb_emp_2_list)
write_keyword_count_column(df_total, 'n_high_comb_emp_3', 'open_ended_3', n_high_comb_emp_3_list)
write_keyword_count_column(df_total, 'n_high_comb_emp_5', 'open_ended_5', n_high_comb_emp_5_list)

df_total['n_high_comb_emp']=df_total['n_high_comb_emp_1']+df_total['n_high_comb_emp_2']+df_total['n_high_comb_emp_3']+df_total['n_high_comb_emp_5']

write_keyword_count_column(df_total, 'e_high_3', 'open_ended_3', e_high_3_list)
write_keyword_count_column(df_total, 'e_high_4', 'open_ended_4', e_high_4_list)
write_keyword_count_column(df_total, 'e_high_5', 'open_ended_5', e_high_5_list)

df_total['e_high_3to5'] = df_total['e_high_3']+df_total['e_high_4']+df_total['e_high_5']

write_keyword_count_column(df_total, 'go_5', 'open_ended_5', go_5_list)
write_keyword_count_column(df_total, 'not_go_5', 'open_ended_5', not_go_5_list)

df_total['go_comb_5']=df_total['go_5']-df_total['not_go_5']

write_keyword_count_column(df_total, 'go_v2', 'open_ended_3', go_v2_3_list)
write_keyword_count_column(df_total, 'not_go_v2', 'open_ended_3', not_go_v2_3_list)

write_keyword_count_column(df_total, 'c_high_2_emp', 'open_ended_2', c_high_2_emp_list)

write_keyword_count_column(df_total, 'c_low_comb_emp_1', 'open_ended_1', c_low_comb_emp_1_list)
write_keyword_count_column(df_total, 'c_low_comb_emp_3', 'open_ended_2', c_low_comb_emp_3_list)
write_keyword_count_column(df_total, 'c_low_comb_emp_4', 'open_ended_3', c_low_comb_emp_4_list)
write_keyword_count_column(df_total, 'c_low_comb_emp_5', 'open_ended_5', c_low_comb_emp_5_list)

df_total['c_low_comb_emp']=df_total['c_low_comb_emp_1']+df_total['c_low_comb_emp_3']+df_total['c_low_comb_emp_4']+df_total['c_low_comb_emp_5']

write_keyword_count_column(df_total, 'c_high_comb_emp_1', 'open_ended_1', c_high_comb_emp_1_list)
write_keyword_count_column(df_total, 'c_high_comb_emp_3', 'open_ended_2', c_high_comb_emp_3_list)
write_keyword_count_column(df_total, 'c_high_comb_emp_4', 'open_ended_3', c_high_comb_emp_4_list)
write_keyword_count_column(df_total, 'c_high_comb_emp_5', 'open_ended_5', c_high_comb_emp_5_list)

df_total['c_high_comb_emp']=df_total['c_high_comb_emp_1']+df_total['c_high_comb_emp_3']+df_total['c_high_comb_emp_4']+df_total['c_high_comb_emp_5']

write_keyword_count_column(df_total, 'e_low_3_emp', 'open_ended_2', e_low_3_emp_list)
write_keyword_count_column(df_total, 'e_high_3_emp', 'open_ended_2', e_high_3_emp_list)

write_keyword_count_column(df_total, 'a_low_1_emp', 'open_ended_1', a_low_emp_1_list)
write_keyword_count_column(df_total, 'a_high_1_emp', 'open_ended_1', a_high_emp_1_list)

write_keyword_count_column(df_total, 'a_low_comb_emp_2', 'open_ended_1', a_low_comb_emp_2_list)
write_keyword_count_column(df_total, 'a_low_comb_emp_3', 'open_ended_2', a_low_comb_emp_3_list)
write_keyword_count_column(df_total, 'a_low_comb_emp_4', 'open_ended_3', a_low_comb_emp_4_list)
write_keyword_count_column(df_total, 'a_low_comb_emp_5', 'open_ended_5', a_low_comb_emp_5_list)

df_total['a_low_comb_emp']=df_total['a_low_comb_emp_2']+df_total['a_low_comb_emp_2']+df_total['a_low_comb_emp_4']+df_total['a_low_comb_emp_5']

write_keyword_count_column(df_total, 'a_high_comb_emp_2', 'open_ended_1', a_high_comb_emp_2_list)
write_keyword_count_column(df_total, 'a_high_comb_emp_3', 'open_ended_2', a_high_comb_emp_3_list)
write_keyword_count_column(df_total, 'a_high_comb_emp_4', 'open_ended_3', a_high_comb_emp_4_list)
write_keyword_count_column(df_total, 'a_high_comb_emp_5', 'open_ended_5', a_high_comb_emp_5_list)

df_total['a_high_comb_emp']=df_total['a_high_comb_emp_2']+df_total['a_high_comb_emp_2']+df_total['a_high_comb_emp_4']+df_total['a_high_comb_emp_5']


#%%
# Create list of features to standardize

zvarlist=['char_count_3',
 'char_count_4',
 'avg_word_1',
 'avg_word_2',
 'avg_word_3',
 'avg_word_4',
 'avg_word_5',
 'not_go_5',
 'go_5',
 'go_comb_5',
 'sum_not',
 'no_5',
 'not_5',
 'n_low_comb_emp',
 'n_high_comb_emp',
 'e_high_3',
 'e_high_4',
 'e_high_5',
 'e_high_3to5',
 'go_v2',
 'not_go_v2',
 'c_high_2_emp',
 'c_low_comb_emp',
 'c_high_comb_emp',
 'e_low_3_emp',
 'e_high_3_emp',
 'a_low_1_emp',
 'a_high_1_emp',
 'a_low_comb_emp',
 'a_high_comb_emp']


#%%
# Standardize list

cols = zvarlist
for col in cols:
    col_zscore = 'Z'+ col
    df_total[col_zscore] = (df_total[col] - df_total[col].mean())/df_total[col].std(ddof=0)


#%%
# Weighting features

df_total['Zno_5']=df_total['Zno_5'] * -1
df_total['Znot_5']=df_total['Znot_5']  *-1
df_total['Zn_low_comb_emp']=df_total['Zn_low_comb_emp']  *-1
df_total['Zc_high_2_emp']=df_total['Zc_high_2_emp']  *1.25
df_total['Zc_low_comb_emp']=df_total['Zc_low_comb_emp']  *-1
df_total['Zc_high_comb_emp']=df_total['Zc_high_comb_emp'] *1.5
df_total['Ze_low_3_emp']=df_total['Ze_low_3_emp']  *-1.25
df_total['Ze_high_3_emp']=df_total['Ze_high_3_emp']  *1.25
df_total['Znot_go_v2']=df_total['Znot_go_v2']  *-1
df_total['Zsum_not']=df_total['Zsum_not']  *-1
df_total['Za_low_1_emp ']=df_total['Za_low_1_emp']  *-1.5
df_total['Za_low_comb_emp']=df_total['Za_low_comb_emp']  *-1
df_total['Za_high_comb_emp']=df_total['Za_high_comb_emp']  *1.5


#%%
df_total['o_pred']=df_total[['Zchar_count_3', 'Zchar_count_4', 'Zno_5', 'Znot_5', 'Ze_high_3to5',  'Zavg_word_2', 
                         'Zavg_word_4', 'Zgo_comb_5']].mean(axis=1)                                


#%%
#Recode o_pred

recode_list=df_total[['o_pred']]

def recode_extreme(predictor_col):
    if predictor_col >=0.75:
        val=.75
    else: 
        val=predictor_col
    return val

for predictor_col in recode_list:
    df_total[predictor_col] = df_total[predictor_col].apply(recode_extreme)


#%%
df_total['n_pred'] = df_total[['Zn_low_comb_emp', 'Zn_high_comb_emp']].mean(axis=1)


#%%
df_total['c_pred'] = df_total[['Zc_high_2_emp','Zc_low_comb_emp','Zc_high_comb_emp','Zavg_word_5']].mean(axis=1)


#%%
df_total['e_pred'] = df_total[['Ze_low_3_emp', 'Ze_high_3_emp', 'Zgo_v2', 'Znot_go_v2']].mean(axis=1)


#%%
df_total['a_pred'] = df_total[['Zavg_word_4', 'Zsum_not', 'Zavg_word_5', 'Za_low_1_emp', 'Za_high_1_emp', 'Za_low_comb_emp','Za_high_comb_emp']].mean(axis=1)

#%% [markdown]
# ##  Machine Learning Section
# 
# Much of the machine learning that we applied did not result in stronger predictions on the public leaderboard compare to the word lists. Therefore, much of these exploratory features have been removed. We retained what was ultimately submitted to the private leaderboard.
# 
# Not all the features that are defined here are important to the prediction. Again, we were lazy and did not prune.

#%%
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

# This function is supposed to count grammatical errors. 
def lang_checker(text):
    tool = language_check.LanguageTool('en-US')
    count=0
    matches = tool.check(text)
    for i in range(len(matches)-1):
        if matches[i].ruleId == 'WHITESPACE_RULE':
            pass
        else:
            count+=1
    return count

def tokenize(text):
    return TextBlob(text).words


#%%
# Compute some spelling-based features
for predictor_col in PREDICTOR_TEXT_COLUMN_NAMES_ALL:
    df_total[predictor_col + "_num_chars"] = df_total[predictor_col].apply(len)
    df_total[predictor_col + "_num_words"] = df_total[predictor_col].apply(word_count)
    df_total[predictor_col + "_num_misspelled"] = df_total[predictor_col].apply(compute_num_spelling_errors)
    df_total[predictor_col + "_flesch_grade"] = df_total[predictor_col].apply(flesch_grade_score) 
    df_total[predictor_col + "_percent_misspelled"] = df_total[[predictor_col + "_num_misspelled",
                              predictor_col + "_num_words"
    ]].apply(lambda x: divide(*x), axis=1)

# Compute readability features
df_total["readability_syllables_count"] = df_total['open_ended_6'].apply(syllables_count) 
df_total["readability_word_count"] = df_total['open_ended_6'].apply(word_count) 
df_total["readability_difficult_count"] = df_total['open_ended_6'].apply(difficult_word_count) 
df_total["readability_sentence_count"] = df_total['open_ended_6'].apply(sentence_count) 
df_total["readability_avg_syllables_per_word"] = df_total['open_ended_6'].apply(avg_syllables_per_word)
df_total["readability_avg_sentence_length"] = df_total['open_ended_6'].apply(avg_sentence_length) 
df_total["readability_flesch_ease_score"] = df_total['open_ended_6'].apply(flesch_ease_score) 
df_total["readability_flesch_grade_score"] = df_total['open_ended_6'].apply(flesch_grade_score) 
df_total["readability_linsear_write_score"] = df_total['open_ended_6'].apply(linsear_write_score) 
df_total["readability_dale_chall_score"] = df_total['open_ended_6'].apply(dale_chall_score) 
df_total["readability_smog_score"] = df_total['open_ended_6'].apply(smog_score) 
df_total["readability_coleman_liau_score"] = df_total['open_ended_6'].apply(coleman_liau_score) 

# Compute variable for the number of grammar errors based on Nick's function. 
df_total["number_grammar_errors"] = df_total['open_ended_6'].apply(lang_checker)
  
# Compute Average Word Length for each open ended comment.     
df_total['Avg_word_length_1']=df_total['open_ended_1_num_chars']/df_total['open_ended_1_num_words']
df_total['Avg_word_length_2']=df_total['open_ended_2_num_chars']/df_total['open_ended_2_num_words']
df_total['Avg_word_length_3']=df_total['open_ended_3_num_chars']/df_total['open_ended_3_num_words']
df_total['Avg_word_length_4']=df_total['open_ended_4_num_chars']/df_total['open_ended_4_num_words']
df_total['Avg_word_length_5']=df_total['open_ended_5_num_chars']/df_total['open_ended_5_num_words']
df_total['Avg_word_length_6']=df_total['open_ended_6_num_chars']/df_total['open_ended_6_num_words']


#%%
# Get list of the numeric columns to paste into the Z-Score variable list
FEATURES = df_total.select_dtypes(include=[np.number]).columns.tolist()
FEATURES


#%%
# Create Z Scores for all new feastures

cols = FEATURES
for col in cols:
    col_zscore = 'Z_'+ col
    df_total[col_zscore] = (df_total[col] - df_total[col].mean())/df_total[col].std(ddof=0)


#%%
Z_Var_List=[
 'Z_open_ended_1_num_words',
 'Z_open_ended_1_num_misspelled',
 'Z_open_ended_1_percent_misspelled',
 'Z_open_ended_2_num_words',
 'Z_open_ended_2_num_misspelled',
 'Z_open_ended_2_percent_misspelled',
 'Z_open_ended_3_num_words',
 'Z_open_ended_3_num_misspelled',
 'Z_open_ended_3_percent_misspelled',
 'Z_open_ended_4_num_words',
 'Z_open_ended_4_num_misspelled',
 'Z_open_ended_4_percent_misspelled',
 'Z_open_ended_5_num_words',
 'Z_open_ended_5_num_misspelled',
 'Z_open_ended_5_percent_misspelled',
 'Z_open_ended_6_num_words',
 'Z_open_ended_6_num_misspelled',
 'Z_open_ended_6_percent_misspelled',
 'Z_O_high_5',
 'Z_C_high_2',
 'Z_A_high_1',
 'Z_E_high_3',
 'Z_A_low_2',
 'Z_A_low_3',
 'Z_A_low_4',
 'Z_A_low_5',
 'Z_N_low_1',
 'Z_N_low_2',
 'Z_N_low_3',
 'Z_N_low_5',
 'Z_C_high_1',
 'Z_C_high_3',
 'Z_C_high_4',
 'Z_C_high_5',
 'Z_A_high_2',
 'Z_A_high_3',
 'Z_A_high_4',
 'Z_A_high_5',
 'Z_N_high_1',
 'Z_N_high_2',
 'Z_N_high_3',
 'Z_N_high_5',
 'Z_O_high_1',
 'Z_O_high_2',
 'Z_O_high_3',
 'Z_O_high_4',
 'Z_E_high_4',
 'Z_E_high_5',
 'Z_A_low_1',
 'Z_E_low_3',
 'Z_GO_3',
 'Z_NOGO_3',
 'Z_GO_5',
 'Z_NOGO_5',
 'Z_NOT_1',
 'Z_NOT_2',
 'Z_NOT_3',
 'Z_NOT_4',
 'Z_NOT_5',
 'Z_NO_5',
 'Z_A_low_comb',
 'Z_N_low_comb',
 'Z_C_high_comb',
 'Z_A_high_comb',
 'Z_N_high_comb',
 'Z_O_high_comb',
 'Z_E_high_3to5',
 'Z_A_not_comb',
 'Z_O_go_comb',
 'Z_open_ended_1_num_chars',
 'Z_open_ended_1_flesch_grade',
 'Z_open_ended_2_num_chars',
 'Z_open_ended_2_flesch_grade',
 'Z_open_ended_3_num_chars',
 'Z_open_ended_3_flesch_grade',
 'Z_open_ended_4_num_chars',
 'Z_open_ended_4_flesch_grade',
 'Z_open_ended_5_num_chars',
 'Z_open_ended_5_flesch_grade',
 'Z_open_ended_6_num_chars',
 'Z_open_ended_6_flesch_grade',
 'Z_readability_syllables_count',
 'Z_readability_word_count',
 'Z_readability_difficult_count',
 'Z_readability_sentence_count',
 'Z_readability_avg_syllables_per_word',
 'Z_readability_avg_sentence_length',
 'Z_readability_flesch_ease_score',
 'Z_readability_flesch_grade_score',
 'Z_readability_linsear_write_score',
 'Z_readability_dale_chall_score',
 'Z_readability_smog_score',
 'Z_readability_coleman_liau_score',
 'Z_number_grammar_errors',
 'Z_Avg_word_length_1',
 'Z_Avg_word_length_2',
 'Z_Avg_word_length_3',
 'Z_Avg_word_length_4',
 'Z_Avg_word_length_5',
 'Z_Avg_word_length_6']


#%%
Z_Var_List


#%%
# Create subset dataframes. 

df_train=df_total.loc[df_total['Source']=='Train'] 
df_test=df_total.loc[df_total['Source']=='Test'] 
df_final=df_total.loc[df_total['Source']=='Final']


#%%
X = df_train[Z_Var_List]
Y = df_train['O_Scale_score']
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.60, test_size=0.40)
O_pipeline_optimizer = TPOTRegressor(generations=20, population_size=20, cv=5,random_state=42, verbosity=2)
O_pipeline_optimizer.fit(X_train,y_train)


#%%
X = df_train[Z_Var_List]
Y = df_train['C_Scale_score']
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.60, test_size=0.40)
C_pipeline_optimizer = TPOTRegressor(generations=20, population_size=20, cv=5,random_state=42, verbosity=2)
C_pipeline_optimizer.fit(X_train,y_train)


#%%
X = df_train[Z_Var_List]
Y = df_train['E_Scale_score']
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.60, test_size=0.40)
E_pipeline_optimizer = TPOTRegressor(generations=20, population_size=20, cv=5,random_state=42, verbosity=2)
E_pipeline_optimizer.fit(X_train,y_train)


#%%
X = df_train[Z_Var_List]
Y = df_train['A_Scale_score']
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.60, test_size=0.40)
A_pipeline_optimizer = TPOTRegressor(generations=20, population_size=20, cv=5,random_state=42, verbosity=2)
A_pipeline_optimizer.fit(X_train,y_train)


#%%
# Auto ML with feature set predicting Agreeableness
X = df_train[Z_Var_List]
Y = df_train['N_Scale_score']
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.60, test_size=0.40)
N_pipeline_optimizer = TPOTRegressor(generations=20, population_size=20, cv=5,random_state=42, verbosity=2)
N_pipeline_optimizer.fit(X_train,y_train)


#%%
# Save predicted values

df_test['O_Scale_pred'] = O_pipeline_optimizer.predict(df_test[Z_Var_List])
df_test['C_Scale_pred'] = C_pipeline_optimizer.predict(df_test[Z_Var_List])
df_test['E_Scale_pred'] = E_pipeline_optimizer.predict(df_test[Z_Var_List])
df_test['A_Scale_pred'] = A_pipeline_optimizer.predict(df_test[Z_Var_List])
df_test['N_Scale_pred'] = N_pipeline_optimizer.predict(df_test[Z_Var_List])

df_final['O_Scale_pred'] = O_pipeline_optimizer.predict(df_final[Z_Var_List])
df_final['C_Scale_pred'] = C_pipeline_optimizer.predict(df_final[Z_Var_List])
df_final['E_Scale_pred'] = E_pipeline_optimizer.predict(df_final[Z_Var_List])
df_final['A_Scale_pred'] = A_pipeline_optimizer.predict(df_final[Z_Var_List])
df_final['N_Scale_pred'] = N_pipeline_optimizer.predict(df_final[Z_Var_List])

df_test.to_csv("ML Test.csv", index = False)
df_final.to_csv("ML Final.csv", index = False)

#%% [markdown]
# ## Deep Learning Section
# 
# We wrote this section to run independently from the previous sections (i.e., there are no independencies). In this way, you can see how we used deep learning without getting confused with all the other junk.

#%%
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub

from keras.layers import Dense, Dropout, Embedding, Flatten, Input, MaxPooling1D
from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasRegressor
from keras import backend as K 
import keras.layers as layers
from keras.models import Model, load_model
from keras.engine import Layer

from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# Initialize session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


#%%
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)


#%%
# Make sure we have a GPU - else this'll take a lifetime or two
sess.list_devices()


#%%
# Make sure directory hierarchy aligns
train_raw_df = pd.read_csv("../materials/Data/siop_ml_train_participant.csv")
df_test = pd.read_csv("../materials/Data/siop_ml_test_participant.csv")
df_dev = pd.read_csv("../materials/Data/siop_ml_dev_participant.csv")


#%%
ATTRIBUTE_LIST = ["E", "A", "O", "C", "N"]

X = train_raw_df[['open_ended_' + str(idx) for idx in range(1, 6)]]
Y = np.array(train_raw_df[[att + "_Scale_score" for att in ATTRIBUTE_LIST]].values)

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=23
)

X_train = [X_train['open_ended_' + str(idx)] for idx in range(1, 6)]
X_test = [X_test['open_ended_' + str(idx)] for idx in range(1, 6)]
X_dev = [df_test['open_ended_' + str(idx)] for idx in range(1, 6)]
X_dev_ = [df_dev['open_ended_' + str(idx)] for idx in range(1, 6)]


#%%
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


#%%
def ElmoRegressionModel(
    dense_dropout_rate=0.5,
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['mse'],
    print_summary=False,
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


#%%
test_scores = []
train_scores = []
estimators = []

ATTRIBUTE_MODEL_PARAMS = [
    dict(dense_dropout_rate=0.7),
    dict(dense_dropout_rate=0.7),
    dict(dense_dropout_rate=0.7),
    dict(dense_dropout_rate=0.7),
    dict(include_hidden_layer=True, dense_dropout_rate=0.2),
]

for idx, att in enumerate(ATTRIBUTE_LIST):
    print("Training for attribute {}".format(att))
    model_params = ATTRIBUTE_MODEL_PARAMS[idx]
    
    clf = KerasRegressor(
        build_fn=lambda: ElmoRegressionModel(**model_params),
        epochs=10,
        batch_size=32,
        verbose=1
    )
    clf.fit(X_train, Y_train[:,idx], validation_data=(X_test, Y_test[:,idx]))
    estimators.append(clf)

    preds_test = clf.predict(X_test)
    preds_train = clf.predict(X_train)
    df_test[att + "_Pred"] = clf.predict(X_dev)
    df_dev[att + "_Pred"] = clf.predict(X_dev_)
    
    pearson_r_test = pearsonr(Y_test[:,idx], preds_test)
    pearson_r_train = pearsonr(Y_train[:,idx], preds_train)
    
    test_scores.append(pearson_r_test)
    train_scores.append(pearson_r_train)
    
    print("{0} - Test r: {1}".format(att, pearson_r_test))
    print("{0} - Train r: {1}".format(att, pearson_r_train))
    print("")
    
print("Average Test r: {}".format(sum([ts[0] for ts in test_scores]) / len(test_scores)))
print("Average Train r: {}".format(sum([ts[0] for ts in train_scores]) / len(train_scores)))


#%%
df_test.to_csv(
    "preds_test_01.csv",
    columns=["Respondent_ID", *[sym + "_Pred" for sym in ATTRIBUTE_LIST]],
    index=False
)

df_dev.to_csv(
    "preds_dev_01.csv",
    columns=["Respondent_ID", *[sym + "_Pred" for sym in ATTRIBUTE_LIST]],
    index=False
)

#%% [markdown]
# ## Winning submission
# 
# Each of the three sets of predicted values generated from the above code were submitted to the private leader board. With the exception of Openness, the best predictors from those were then averaged together to form a fourth submission. Our openness predictor was poor, so we continued to tinker with it on the fourth submission. In all cases the averaged values had stronger correlations than the independent values.
# 
# The final submission was as follows:
# - Openness: Word List
# - Concientiousness: averaged the z-transformed predicted values from World List and Deep Learning
# - Agreeableness: averaged the z-transformed predicted values from Machine Learning and Deep Learning
# - Extraversion: averaged the z-transformed predicted values from Word List and Deep Learning
# - Neuroticism: averaged the z-transformed predicted values from Word List and Deep Learning

#%%



