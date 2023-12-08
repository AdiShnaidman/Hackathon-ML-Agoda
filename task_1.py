import numpy as np
import pandas as pd
import plotly.express as px
import warnings

from sklearn.svm import SVC

warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score


def kfold_cross_validation(model, X, y,kernel=None, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    f1_scores = []
    accuracy_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if kernel is not None:
            model.set_params(kernel=kernel)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        f1 = f1_score(y_test, y_pred, average='macro')
        accuracy = accuracy_score(y_test, y_pred)

        f1_scores.append(f1)
        accuracy_scores.append(accuracy)

    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    mean_accuracy = np.mean(accuracy_scores)
    std_accuracy = np.std(accuracy_scores)

    print(f"Mean F1 Score: {mean_f1:.4f}")
    print(f"Standard Deviation of F1 Score: {std_f1:.4f}")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Standard Deviation of Accuracy: {std_accuracy:.4f}")
    return mean_f1, mean_accuracy, std_f1, std_accuracy


def parse_data(file_path):
    data = pd.read_csv(file_path)

    X = data.drop('cancellation_datetime', axis=1)
    y = data['cancellation_datetime']

    X = X.values
    y = y.values

    return X, y


def task_1(file_path):
    X, y = parse_data(file_path)
    models = [
        AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=200, random_state=42),
        LogisticRegression(max_iter=5000, C=1.0),  # Adjust C parameter for Logistic Regression
        DecisionTreeClassifier(max_depth=5),  # Adjust max_depth parameter for Decision Tree
        RandomForestClassifier(n_estimators=100, max_depth=10),  # Adjust hyperparameters for Random Forest
    ]

    for model in models:
        print(f"Model: {model.__class__.__name__}")
        kfold_cross_validation(model, X, y)  # Use F1 score as the evaluation metric
        print("============================")
