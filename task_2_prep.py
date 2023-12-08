import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from joblib import dump, load


import preprocess_train2


def kfold_cross_validation2(model, X, y, kernel=None, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if kernel is not None:
            model.set_params(kernel=kernel)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.where(y_pred < 0, -1, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        rmse_scores.append(rmse)

    mean_rmse1 = np.mean(rmse_scores)
    std_rmse1 = np.std(rmse_scores)

    print(f"Mean RMSE: {mean_rmse1:.4f}")
    print(f"Standard Deviation of RMSE: {std_rmse1:.4f}")
    return mean_rmse1, std_rmse1


def remove_outliers_zscore(data, column, threshold=3):
    column_data = data[column]
    z_scores = np.abs((column_data - np.mean(column_data)) / np.std(column_data))
    return data[z_scores < threshold]


# Usage

def parse_data2(data):
    data = remove_outliers_zscore(data, 'original_selling_amount')
    # data = data.drop(["request_latecheckin","request_airport","days_ahead","request_nonesmoke","no_of_extra_bed","same_day_booking_checkin","VND","hotel_star_rating","request_earlycheckin","credit","is_user_logged_in"], axis=1)
    X = data.drop(["original_selling_amount"], axis=1)  # Features
    y = data["original_selling_amount"]  # Target variable

    X = X.values
    y = y.values

    return X, y


def task_2_prep(df):
    X, y = parse_data2(df)
    ridge_model = Ridge(alpha=0.1)

    # Perform cross-validation with 5 folds
    cv_scores = cross_val_score(ridge_model, X, y, cv=5, scoring='neg_mean_squared_error')
    ridge_model = ridge_model.fit(X, y)
    dump(ridge_model, 'ridge_model.pkl')


if __name__ == "__main__":
    df = preprocess_train2.preprocess2(pd.read_csv("agoda_cancellation_train.csv"), True)
    task_2_prep(df)
