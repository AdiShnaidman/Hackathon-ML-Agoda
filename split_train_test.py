from sklearn.model_selection import train_test_split
import pandas as pd
def main():
    df = pd.read_csv('agoda_cancellation_train.csv')
    train=df.sample(frac=0.7, random_state=200)
    test=df.drop(train.index)
    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)


if __name__ == "__main__":
    #main()
    pass