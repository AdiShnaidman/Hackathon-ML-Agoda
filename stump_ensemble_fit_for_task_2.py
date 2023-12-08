import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from joblib import dump, load

# Assuming 'df' is your DataFrame with features and target variable
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('clean_train1_for2.csv')

# Separate the features and target variable
X = df.drop('cancellation_datetime', axis=1)  # Features
y = df['cancellation_datetime']  # Target variable

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Create decision stump classifier as the base estimator
base_estimator = DecisionTreeClassifier(max_depth=2)

# Initialize lists to store F1 loss and ensemble size
f1_losses = []
ensemble_sizes = []

# Iterate over increasing ensemble sizes from 10 to 100

# Create the ensemble using AdaBoost with decision stump as the base estimator
ensemble = AdaBoostClassifier(estimator=base_estimator, n_estimators=500, random_state=42)

# Fit the ensemble model to the training data
ensemble.fit(X_train, y_train)

# Make predictions on the training data
y_train_pred = ensemble.predict(X_train)

# Calculate the F1 score
f1 = f1_score(y_train, y_train_pred, average='macro')

# Calculate the F1 loss (invert the F1 score)
f1_loss = 1 - f1

# Append the F1 loss to the list
f1_losses.append(f1_loss)

# save ensemble args
dump(ensemble, 'estimator1_for2.joblib')


