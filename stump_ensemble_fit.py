import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from joblib import dump, load
from preprocess_train import preprocess
from sklearn.tree import DecisionTreeClassifier

df = preprocess('agoda_cancellation_train.csv', 0)

# Separate the features and target variable
X = df.drop('cancellation_datetime', axis=1)  # Features
y = df['cancellation_datetime']  # Target variable

# Create decision stump classifier as the base estimator
base_estimator = DecisionTreeClassifier(max_depth=2)

# Create the ensemble using AdaBoost with decision stump as the base estimator
ensemble = AdaBoostClassifier(estimator=base_estimator, n_estimators=500, random_state=42)

# Fit the ensemble model to the training data
ensemble.fit(X, y)

# save ensemble args
dump(ensemble, 'estimator.joblib')