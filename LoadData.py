import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

random_seed = 8424  # this is very important!

data = pd.read_csv("training_data_participant/siop_ml_train_participant.csv")

X = data.iloc[:, 1:6]
y = data.iloc[:, 6:11]

# rename columns for easier access
X.rename(columns={'open_ended_1': 'A',
                  'open_ended_2': 'C',
                  'open_ended_3': 'E',
                  'open_ended_4': 'N', 
                  'open_ended_5': 'O'}, inplace=True)

y.rename(columns={'E_Scale_score': 'E',
                  'A_Scale_score': 'A',
                  'O_Scale_score': 'O',
                  'C_Scale_score': 'C',
                  'N_Scale_score': 'N'}, inplace=True)

# put all five responses into one "paragraph"
X = X.stack().groupby(level=0).apply(' '.join)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    random_state=random_seed,
                                                    test_size=0.1,
                                                    shuffle=True)

model = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 3))),
    ('tfidf', TfidfTransformer(use_idf=False)),
    ('svd', TruncatedSVD(random_state=random_seed, n_components=80)),
    ('model', SVR(kernel='linear'))
])

sum_r = 0

for trait in ['O', 'C', 'E', 'A', 'N']:
    model.fit(X_train, y_train[trait])
    y_pred = model.predict(X_test)
    r = np.corrcoef(y_pred, y_test[trait])[0, 1] 
    print((trait, r))
    sum_r += r

print(('Mean r:', sum_r/5))

model = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer(use_idf=False)),
    ('svd', TruncatedSVD(random_state=random_seed, n_components=3)),
    ('model', LinearRegression())
])

sum_r = 0

for trait in ['O', 'C', 'E', 'A', 'N']:
    model.fit(X_train, y_train[trait])
    y_pred = model.predict(X_test)
    r = np.corrcoef(y_pred, y_test[trait])[0, 1] 
    print((trait, r))
    sum_r += r

print(('Mean r:', sum_r/5))