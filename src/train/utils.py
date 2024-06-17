# utils.py

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from itertools import product
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import OneHotEncoder

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

# wrapper feature selection evaluates a new model for every feature combination
# the computation cost therefore is Order n^2
def wrapper_find_best_features(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    ''' Uses wrapper feature selection to find the feature selection with the highest accuracy'''

    # determine the number of columns
    n_cols = X.shape[1] # 5
    best_subset, best_score = None, 0.0

    # enumerate all combinations of input features
    # product() here returns all combinations of True and False of length 5
    for subset in product([True, False], repeat=n_cols):
    
        # convert into column indexes: [False, True] -> [0, 1] for example
        ix = [i for i, x in enumerate(subset) if x]
    
        # if the sequence has no column indexes (all False) we can skip that sequence
        if len(ix) == 0:
            continue
        
        # select the column indexes to choose the columns in the dataset
        X_new = X.to_numpy()[:, ix]
    
        # evaluate this subset of the dataset
    
        # define the model
        model = DecisionTreeClassifier()

        # cross validation procedure
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state =1)

        # evaluate model
        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=1)
    
        # summarise scores
        result = np.mean(scores)
    
        # check if better than the best score so far
        if best_score is None or result >= best_score:
            # update best score (and subset)
            best_subset, best_score = ix, result
    
        return best_subset
    
# filter feature selection uses statistical techniques to evaluate the relationship between each feature and the target
def filter_find_best_features():
    '''Uses filter feature selection to find the feature selection with the highest accuracy'''
    pass