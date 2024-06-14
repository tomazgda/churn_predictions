import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

def clean_and_split(filename: str, features: list[str], test_size: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """reads a csv into a dataframe, removes rows with na targets, and splits data"""
    # read csv into dataframe
    data = pd.read_csv(filename)

    # write the first 10 rows of the dataframe into a new csv file
    data.head(10).to_csv("telco_data_head.csv")

    # remove rows with missing targets
    data_without_na = data.dropna(axis=0, subset=['Churn'], inplace=False)

    # seperate target from features, and encode target data
    y = np.where(data_without_na['Churn'] == 'Yes', 1, 0)

    # drop target from whole data set, return features which will be called X
    X = data_without_na.drop(['Churn'], axis=1, inplace=False)[features]

    # seperature training from validation data
    return train_test_split(X, y, test_size=test_size, random_state=0)

