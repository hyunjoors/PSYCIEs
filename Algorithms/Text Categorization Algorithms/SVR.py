import LoadData

# SVR

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


# Regression
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

