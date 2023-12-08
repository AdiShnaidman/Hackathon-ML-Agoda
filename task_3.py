from joblib import load
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from preprocess_train import preprocess

def task_3():

    X_train = preprocess('train.csv', 0)
    y_train = X_train['cancellation_datetime']
    X_train = X_train.loc[:, ['days_ahead', '>30', '<1','2-3']]

    base_estimator = DecisionTreeClassifier(max_depth=2)
    ensemble = AdaBoostClassifier(estimator=base_estimator, n_estimators=500, random_state=42)
    # Define the percentage values
    percentages = np.arange(10, 100, 9)

    train_f1_scores = []
    test_f1_scores = []

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3)
    for percentage in percentages:
        n_samples = int((percentage/100) * len(y_train))

        X_train_subset = X_train[:n_samples]
        y_train_subset = y_train[:n_samples]

        ensemble.fit(X_train_subset, y_train_subset)
        y_train_pred = ensemble.predict(X_train)
        y_test_pred = ensemble.predict(X_test)

        # Calculate the F1 scores
        train_f1 = f1_score(y_train, y_train_pred, average='macro')
        test_f1 = f1_score(y_test, y_test_pred, average='macro')

        # Append the scores to the lists
        train_f1_scores.append(train_f1)
        test_f1_scores.append(test_f1)


    #Graph 1
    plt.plot(percentages, train_f1_scores, label='Train')
    plt.plot(percentages, test_f1_scores, label='Test')
    plt.xlabel('Percentage of Training Data')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Percentage of Training Data')
    plt.legend()
    plt.show()

    ensemble = load('hackathon_code/estimator.joblib')

    # Calculate the total sum of all values
    total_sum = sum(ensemble.feature_importances_)
    threshold = 0.03 * total_sum

    filtered_values = [value for value in ensemble.feature_importances_ if value > threshold]
    filtered_labels = [label for value, label in zip(ensemble.feature_importances_, ensemble.feature_names_in_) if value > threshold]

    # Calculate the sum of the values that passed the threshold
    filtered_sum = sum(filtered_values)

    # Append an "Other" label and its corresponding sum
    filtered_labels.append("Other")
    filtered_values.append(total_sum - filtered_sum)

    # Calculate the threshold value (e.g., 10% of the total sum)
    threshold = 0.1 * total_sum
    plt.pie(filtered_values, labels=filtered_labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Features\' Influence on Prediction')
    plt.show()


if __name__ == "__main__":
    task_3()