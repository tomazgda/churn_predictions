# utils.py

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

def clean_and_split(data: pd.DataFrame, features: list[str], test_size: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''reads a csv into a dataframe, removes rows with na targets, and splits data'''
    
    # remove rows with missing targets
    data_without_na = data.dropna(axis=0, subset=['Churn'], inplace=False)

    # seperate target from features, and encode target data
    y = np.where(data_without_na['Churn'] == 'Yes', 1, 0)

    # drop target from whole dataset, return features which will be called X
    X = data_without_na.drop(['Churn'], axis=1, inplace=False)[features]

    # seperature training from validation data
    return train_test_split(X, y, test_size=test_size, random_state=0)

