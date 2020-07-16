%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import os

# path to read in csv

X = voice.drop("label", axis=1)
y = voice["label"]
print(X.shape, y.shape)

#splitting trained and tested

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)



from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier

 LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

 print(f"Training Data Score: {classifier.score(X_train, y_train)}")
print(f"Testing Data Score: {classifier.score(X_test, y_test)}")

#read prediction of loan status based off of tested data

predictions = classifier.predict(X_test)
print(f"First 10 Predictions:   {predictions[:10]}")
print(f"First 10 Actual labels: {y_test[:10].tolist()}")

pd.DataFrame({"Prediction": predictions, "Actual": y_test}).reset_index(drop=True)