from joblib import load
import pandas as pd
from hackathon_code import preprocess1_task2
from hackathon_code import preprocess_train2




def task_2(filename):
    #run model2
    df = pd.read_csv(filename)
    X_test_1 = preprocess_train2.preprocess2(df, False)
    ridge = load('hackathon_code/ridge_model.pkl')
    pred = ridge.predict(X_test_1)
    id = df['h_booking_id']
    res = pd.DataFrame({'ID': id, 'predicted_selling_amount': pred})

    ensemble = load('hackathon_code/estimator1_for2.joblib')
    cols_when_model_builds = ensemble.feature_names_in_
    df["original_selling_amount"] = pred
    X_test = preprocess1_task2.preprocess1_2(df)
    X_test = X_test[cols_when_model_builds]
    pred_label = ensemble.predict(X_test)
    for index, row in res.iterrows():
        if pred_label[index] == 0:
            res.at[index, 'predicted_selling_amount'] = -1
    res.to_csv("agoda_cost_of_cancellation.csv", index=False)
