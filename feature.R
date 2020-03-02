library(caret)
library(data.table)
library(dplyr)
library(ggpubr)
library(glmnet)
library(koRpus)
library(ldatuning)
library(leaps)
library(magrittr)
library(neuralnet)
# devtools::install_github("kbenoit/quanteda.dictionaries")
library(quanteda.dictionaries)
library(quanteda)
library(randomForest)
library(readr)# to read in csv
library(SentimentAnalysis)
library(SnowballC) # for stemming
library(stopwords)
library(text2vec)#for word vectors
library(textmineR)
library(textstem)#for lemmetization
library(tidytext) # tidy implimentation of NLP methods
library(tidyverse) # general utility & workflow functions
library(tm) # general text mining functions, making document term matrixes
library(topicmodels) # for LDA topic modelling 

# soi.data == df

# 1. Import & Combine Data ----------------------------------------------------
# Import raw data
df_train <- read.csv("full_data/siop_ml_train_participant.csv", stringsAsFactors = F)
df_dev <- read.csv("full_data/siop_ml_dev_participant.csv", stringsAsFactors = F)
df_test <- read.csv("full_data/siop_ml_test_participant.csv", stringsAsFactors = F)
df <- read.csv("full_data/2019_siop_ml_comp_data.csv", stringsAsFactors = F)

# Concatenate the 5 text responses into one column of text
paste_wo_NA <- function(x){
  not_NA <- ifelse(!is.na(x), TRUE, FALSE)
  new_str <- x[not_NA]
  return(paste(new_str, collapse = " "))
}
df$all_text <- apply(df[, paste0("open_ended_", 1:5)], 1, paste_wo_NA)


# 2. Topic Modeling -----------------------------------------------------------
dtms <- lda_metrics <- lda_models <- lda_probs <- list()

# Store names of text variables
text_vars <- c(paste0("open_ended_", 1:5), "all_text")

for(i in text_vars){
  # Create document-term matrices
  dtms[[i]] <- dfm(corpus(df, text_field = i),
                    tolower = TRUE,
                    remove = stopwords('english'),
                    remove_punct = TRUE,
                    remove_numbers = TRUE,
                    remove_symbols = TRUE,
                    remove_url = FALSE,
                    stem = FALSE,
                    ngrams = 1:5)
  
  # Remove tokens that occur in only 1 document
  dtms[[i]] <- dfm_trim(dtms[[i]], min_docfreq = 2)
  
  # Gather metrics to decide how many topics to extract
  lda_metrics[[i]] <- FindTopicsNumber(dtms[[i]],
                                        topics = seq(2, 75, 1),
                                        metrics = c("Griffiths2004", "CaoJuan2009"),
                                        method = "Gibbs",
                                        control = list(seed = 77),
                                        mc.cores = parallel::detectCores() - 1)
}

# Identify optimal number of topics to extract (k)...
metrics <- lapply(lda_metrics, function(x){
  return(c(x$topics[which.max(x$Griffiths2004)], x$topics[which.min(x$CaoJuan2009)]))
}) %>% do.call(rbind, .) %>% set_colnames(c("Griffiths", "CaoJuan"))

# ...and also inspect lda_metrics visually
# FindTopicsNumber_plot(lda_metrics$open_ended_1)
# FindTopicsNumber_plot(lda_metrics$open_ended_2)
# FindTopicsNumber_plot(lda_metrics$open_ended_3)
# FindTopicsNumber_plot(lda_metrics$open_ended_4)
# FindTopicsNumber_plot(lda_metrics$open_ended_5)
# FindTopicsNumber_plot(lda_metrics$all_text)

# Decided on the following number of topics (k) to extract for each SJT item
opt_k <- c(
   45  # open_ended_1
  ,36  # open_ended_2
  ,42  # open_ended_3
  ,51  # open_ended_4
  ,60  # open_ended_5
  ,26  # all_text
) %>% set_names(text_vars)

# Run and store LDA topic models
for(i in text_vars) {
  lda_models[[i]] <- LDA(dtms[[i]],
                        k = opt_k[i],
                        method = "Gibbs",
                        control = list(seed = 77))
  
  # Extract posterior probabilities
  lda_probs[[i]] <- topicmodels::posterior(lda_models[[i]])$topics
  
  # Name topics according to the questions they came from
  if(which(text_vars == i) <= 5) {
    question <- paste0("OE", which(text_vars == i))
  } else {
    question <- "all_text"
  }
  colnames(lda_probs[[i]]) <- paste0("topic_", 1:ncol(lda_probs[[i]]), "_", question)
}

# Combine all posterior probabilities in one dataframe
lda_probs_agg <- lda_probs %>% do.call(cbind, .)


# 3. Dictionary Lookups -------------------------------------------------------

dictionary_list <- list(
  data_dictionary_AFINN
  ,data_dictionary_geninqposneg
  ,data_dictionary_HuLiu
  ,data_dictionary_LaverGarry
  ,data_dictionary_LoughranMcDonald
  ,data_dictionary_LSD2015
  ,data_dictionary_MFD
  ,data_dictionary_NRC
  ,data_dictionary_RID
  ,data_dictionary_sentiws
  ,data_dictionary_uk2us
)

# Only keep these variables for the first dictionary lookup
dont_repeat <- c("docname", "Segment", "WC", "WPS", "Sixltr", "Dic", 
                 "AllPunc", "Period", "Comma", "Colon", "SemiC", "QMark",
                 "Exclam", "Dash", "Quote", "Apostro", "Parenth", "OtherP")

for(i in 1:length(dictionary_list)){
  if(i == 1){  # i.e., keep the 'dont_repeat' variables above
    dictionaries <- liwcalike(df$all_text, dictionary_list[[i]])
    dictionaries <- 
      dictionaries[,-which(colnames(dictionaries) %in% c("docname","Segment"))]
  } else {  # i.e., do not keep the 'dont_repeat' variables above
    tmp <- liwcalike(df$all_text, dictionary_list[[i]])
    dictionaries <- cbind(
      dictionaries
      ,tmp[,-which(colnames(tmp) %in% dont_repeat)]
    )
  }
}

# Name the dictionary variables
colnames(dictionaries) <- 
  paste0("dict_", make.unique(colnames(dictionaries), sep = "."))


# 4. Sentiment Analysis -------------------------------------------------------
# Sentiment: Approximate the sentiment (polarity) of text by sentence. This function allows the user to easily
# alter (add, change, replace) the default polarity an valence shifters dictionaries to suit the context
# dependent needs of a particular data set. See the polarity_dt and valence_shifters_dt arguments for more information. Other hyper-parameters may add additional fine tuned control of the
# algorithm that may boost performance in different contexts.


# Single sentiment score that accounts for negation, amplification, etc.
# sentiment_by: Polarity Score (Sentiment Analysis) By Groups
# Approximate the sentiment (polarity) of text by grouping variable(s). For a full description of the
# sentiment detection algorithm see sentiment. See sentiment for more details about the algorithm,
# the sentiment/valence shifter keys that can be passed into the function, and other arguments that can
# be passed.
# . ave_sentiment - Sentiment/polarity score mean average by grouping variable
sentiment <- apply(df[, text_vars], 2, function(x){sentimentr::sentiment_by(sentimentr::get_sentences(x))$ave_sentiment}) %>% 
    set_colnames(c(paste0("sent", 1:5), "sent_all_text"))

# Emotion specific scores (e.g., joy, fear, surprise, etc.)
nrc_sentiment <- syuzhet::get_nrc_sentiment(df$all_text)
colnames(nrc_sentiment) <- paste0("nrc_", colnames(nrc_sentiment))


all.corpus <- Corpus(VectorSource(df$all_text))
q1.corpus <- Corpus(VectorSource(df$open_ended_1))
q2.corpus <- Corpus(VectorSource(df$open_ended_2))
q3.corpus <- Corpus(VectorSource(df$open_ended_3))
q4.corpus <- Corpus(VectorSource(df$open_ended_4))
q5.corpus <- Corpus(VectorSource(df$open_ended_5))

# analyzeSentiment(x, language = "english",
# aggregate = NULL, rules = defaultSentimentRules(),
# removeStopwords = TRUE, stemming = TRUE, ...)
sentiment.all <- analyzeSentiment(all.corpus)
sentiment.q1 <- analyzeSentiment(q1.corpus)
sentiment.q2 <- analyzeSentiment(q2.corpus)
sentiment.q3 <- analyzeSentiment(q3.corpus)
sentiment.q4 <- analyzeSentiment(q4.corpus)
sentiment.q5 <- analyzeSentiment(q5.corpus)

sentiment.all.df <- data.frame(sentiment.all)
sentiment.q1.df <- data.frame(sentiment.q1)
sentiment.q2.df <- data.frame(sentiment.q2)
sentiment.q3.df <- data.frame(sentiment.q3)
sentiment.q4.df <- data.frame(sentiment.q4)
sentiment.q5.df <- data.frame(sentiment.q5)

colnames(sentiment.all.df) <- paste("all", colnames(sentiment.q1.df), sep ="_")
colnames(sentiment.q1.df) <- paste("q1", colnames(sentiment.q1.df), sep ="_")
colnames(sentiment.q2.df) <- paste("q2", colnames(sentiment.q2.df), sep ="_")
colnames(sentiment.q3.df) <- paste("q3", colnames(sentiment.q3.df), sep ="_")
colnames(sentiment.q4.df) <- paste("q4", colnames(sentiment.q4.df), sep ="_")
colnames(sentiment.q5.df) <- paste("q5", colnames(sentiment.q5.df), sep ="_")


# 5. Readability Indices-------------------------------------------------------
# Coleman Liau short
# Formula includes average word length, # of sentences, and # of words
readability_CL <- apply(df[, paste0("open_ended_", 1:5)], 2, function(x){
  textstat_readability(x, measure = "Coleman.Liau.short")})
readability_CL <- do.call(cbind,readability_CL[paste0("open_ended_", 1:5)]) %>%
  select(2,4,6,8,10) %>%
  `colnames<-`(c(paste0("readability_CL", 1:5)))

# Danielson Bryan 2
# Forumla includes # of characters, # of blanks, and number of sentences
readability_DB <- apply(df[, paste0("open_ended_", 1:5)], 2, function(x){
  textstat_readability(x, measure = "Danielson.Bryan")}) 
readability_DB <- do.call(cbind,readability_DB[paste0("open_ended_", 1:5)]) %>%
  select(2,4,6,8,10) %>%
  `colnames<-`(c(paste0("readability_DB", 1:5)))

# readability <- textstat_readability(soi.data$all_text, measure = c("Flesch.Kincaid", 
#                                                                    "Dale.Chall.old",
#                                                                    "Wheeler.Smith", 
#                                                                    "meanSentenceLength",
#                                                                    "meanWordSyllables",
#                                                                    "Strain",
#                                                                    "SMOG",
#                                                                    "Scrabble",
#                                                                    "FOG",
#                                                                    "Farr.Jenkins.Paterson",
#                                                                    "DRP",
#                                                                    "Dale.Chall")) 

# There is no "soi.data", guessing soi.data is just df

readability <- textstat_readability(df$all_text, measure = c("Flesch.Kincaid", 
                                                             "Dale.Chall.old",
                                                             "Wheeler.Smith", 
                                                             "meanSentenceLength",
                                                             "meanWordSyllables",
                                                             "Strain",
                                                             "SMOG",
                                                             "Scrabble",
                                                             "FOG",
                                                             "Farr.Jenkins.Paterson",
                                                             "DRP",
                                                             "Dale.Chall")) 

# 6. Lexical Diversity --------------------------------------------------------
dtms_nzv <- list()

for(i in 1:5){
  
  dtms_nzv[[i]] <-dfm(
    corpus(df, text_field = paste0("open_ended_", i))
    ,tolower=TRUE
    ,remove_punct=TRUE
    ,remove_numbers=TRUE
    ,remove_symbols=TRUE
    ,ngrams = 1:5
  ) %>% dfm_tfidf(.)
  
  nzv_index <- caret::nearZeroVar(as.matrix(dtms_nzv[[i]]))
  dtms_nzv[[i]] <- dtms_nzv[[i]][,-nzv_index]
  colnames(dtms_nzv[[i]]) <- paste0(colnames(dtms_nzv[[i]]), "_OE", i)
  
}

dtm_agg <- dtms_nzv %>% do.call(cbind, .)

# These indices use # of unique tokens and # of total tokens
lex_div <- quanteda::textstat_lexdiv(dtm_agg, measure = c("C", "R", "D"))


# 7. DTM --------------------------------------------------------

soi.data <- df # seems like soi.data is df

prep_fun <- tolower
tok_fun <- word_tokenizer
#all qs
all.it_train <- itoken(soi.data$all_text,
                       preprocessor = prep_fun,
                       tokenizer = tok_fun,
                       ids = soi.data$Respondent_ID,
                       progressbar = TRUE)
all.vocab <- create_vocabulary(all.it_train)
all.vocab <- prune_vocabulary(all.vocab, term_count_min = 25, doc_proportion_max = 0.90)#worked with 25/.90
all.vectorizer <- vocab_vectorizer(all.vocab)
all.dtm <- create_dtm(all.it_train, all.vectorizer)
all.dtm.df <- data.frame(as.matrix(all.dtm)) #convert dtm to dataframe
colnames(all.dtm.df) <- paste("all", colnames(all.dtm.df), sep ="_")
#q1
q1.it_train <- itoken(soi.data$open_ended_1,
                      preprocessor = prep_fun,
                      tokenizer = tok_fun,
                      ids = soi.data$Respondent_ID,
                      progressbar = TRUE)
q1.vocab <- create_vocabulary(q1.it_train)
q1.vocab <- prune_vocabulary(q1.vocab, term_count_min = 25, doc_proportion_max = 0.90)
q1.vectorizer <- vocab_vectorizer(q1.vocab)
q1.dtm <- create_dtm(q1.it_train, q1.vectorizer)
q1.dtm.df <- data.frame(as.matrix(q1.dtm)) #convert dtm to dataframe
colnames(q1.dtm.df) <- paste("q1", colnames(q1.dtm.df), sep ="_")
#q2
q2.it_train <- itoken(soi.data$open_ended_2,
                      preprocessor = prep_fun,
                      tokenizer = tok_fun,
                      ids = soi.data$Respondent_ID,
                      progressbar = TRUE)
q2.vocab <- create_vocabulary(q2.it_train)
q2.vocab <- prune_vocabulary(q2.vocab, term_count_min = 25, doc_proportion_max = 0.90)
q2.vectorizer <- vocab_vectorizer(q2.vocab)
q2.dtm <- create_dtm(q2.it_train, q2.vectorizer)
q2.dtm.df <- data.frame(as.matrix(q2.dtm)) #convert dtm to dataframe
colnames(q2.dtm.df) <- paste("q2", colnames(q2.dtm.df), sep ="_")
#q3
q3.it_train <- itoken(soi.data$open_ended_3,
                      preprocessor = prep_fun,
                      tokenizer = tok_fun,
                      ids = soi.data$Respondent_ID,
                      progressbar = TRUE)
q3.vocab <- create_vocabulary(q3.it_train)
q3.vocab <- prune_vocabulary(q3.vocab, term_count_min = 25, doc_proportion_max = 0.90)
q3.vectorizer <- vocab_vectorizer(q3.vocab)
q3.dtm <- create_dtm(q3.it_train, q3.vectorizer)
q3.dtm.df <- data.frame(as.matrix(q3.dtm)) #convert dtm to dataframe
colnames(q3.dtm.df) <- paste("q3", colnames(q3.dtm.df), sep ="_")
#q4
q4.it_train <- itoken(soi.data$open_ended_4,
                      preprocessor = prep_fun,
                      tokenizer = tok_fun,
                      ids = soi.data$Respondent_ID,
                      progressbar = TRUE)
q4.vocab <- create_vocabulary(q4.it_train)
q4.vocab <- prune_vocabulary(q4.vocab, term_count_min = 25, doc_proportion_max = 0.90)
q4.vectorizer <- vocab_vectorizer(q4.vocab)
q4.dtm <- create_dtm(q4.it_train, q4.vectorizer)
q4.dtm.df <- data.frame(as.matrix(q4.dtm)) #convert dtm to dataframe
colnames(q4.dtm.df) <- paste("q4", colnames(q4.dtm.df), sep ="_")
#q5
q5.it_train <- itoken(soi.data$open_ended_5,
                      preprocessor = prep_fun,
                      tokenizer = tok_fun,
                      ids = soi.data$Respondent_ID,
                      progressbar = TRUE)
q5.vocab <- create_vocabulary(q5.it_train)
q5.vocab <- prune_vocabulary(q5.vocab, term_count_min = 25, doc_proportion_max = 0.90)
q5.vectorizer <- vocab_vectorizer(q5.vocab)
q5.dtm <- create_dtm(q5.it_train, q5.vectorizer)
q5.dtm.df <- data.frame(as.matrix(q5.dtm)) #convert dtm to dataframe
colnames(q5.dtm.df) <- paste("q5", colnames(q5.dtm.df), sep ="_")


# 8. Misc. Feature Engineering ------------------------------------------------

# Average number of characters per word
chars <- apply(df[, paste0("open_ended_", 1:5)], 2, nchar) 
words <- apply(df[, paste0("open_ended_", 1:5)], 2, quanteda::ntoken)
avg_token_length <- (chars / words)
avg_token_length <- 
  cbind(avg_token_length, rowMeans(chars) / rowMeans(words)) %>%
  set_colnames(c(paste0("char_per_word", 1:5), "char_per_word_all_text"))

# Spelling errors
errors <- unlist(lapply(hunspell::hunspell(df$all_text), length))

# Profanity
profanity <- sentimentr::profanity_by(df$all_text)$profanity_count


# 9. Combine Predictors -------------------------------------------------------
X <- cbind(
  lda_probs_agg
  ,dictionaries
  ,sentiment
  ,nrc_sentiment
  ,readability_CL
  ,readability_DB
  ,lex_div_R = lex_div$R  # Guirad's Root TTR
  ,lex_div_C = lex_div$C  # Herdan's C
  ,lex_div_D = lex_div$D  # Simpson's D
  ,avg_token_length
  ,errors
  ,profanity
  ,sentiment.all.df
  ,sentiment.q1.df, sentiment.q2.df, 
                  sentiment.q3.df, sentiment.q4.df, sentiment.q5.df, readability,
                  all.dtm.df, q1.dtm.df, q2.dtm.df, q3.dtm.df, q4.dtm.df, q5.dtm.df
)

#export mega-dataset
write.csv(X, "mega_dataset.csv")
