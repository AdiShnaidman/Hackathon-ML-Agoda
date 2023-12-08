from joblib import load
import pandas as pd

from hackathon_code import preprocess_train

def task_1(filename):
    ensemble = load('hackathon_code/estimator.joblib')
    cols_when_model_builds = ensemble.feature_names_in_
    X_test = preprocess_train.preprocess(filename, 1)
    X_test_1 = X_test.drop('h_booking_id',axis=1)
    X_test_1 = X_test_1[cols_when_model_builds]
    pred = ensemble.predict(X_test_1)
    id = X_test['h_booking_id']
    res= pd.DataFrame({'ID': id , 'cancellation':pred})
    res.to_csv("agoda_cancellation_prediction.csv", index=False)



if __name__ == "__main__":
    task_1('Agoda_Test_1.csv')







