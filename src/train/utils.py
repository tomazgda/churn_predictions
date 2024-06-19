# utils.py

import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from itertools import product


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

# Everything beyond this point is untested!!!
# -------------------------------------------

# wrapper feature selection evaluates a new model for every feature combination
# the computation cost therefore is Order 2^n
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
    
        # cross validation procedure
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state =1)

        # evaluate model
        scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=1)
    
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

def create_preprocessor(X: pd.DataFrame, subset: np.ndarray) -> ColumnTransformer:
    '''take a feature subset, and return a preprocessor'''

    # series descripting of columns are of dtype object
    L = (X.dtypes == 'object')

    # Obtain a list of all columns containing categorical variables in the training data
    categorical_cols = list(L[L].index)

    # Create a list of all columns containg numerical variables in the training data 
    numerical_cols = [e for e in L.index if not (e in L[L].index)]
    
    # define the transformer for numerical data
    numerical_transformer = SimpleImputer(strategy='mean')

    # define the transformer for categoric data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Column Transformer bundles together multiple transformation stages
    preprocessor = ColumnTransformer(
        transformers = [
            ('num', numerical_transformer, numerical_cols), # need to define numerical_cols and categorical_cols
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    return preprocessor

def objective(X: pd.DataFrame, y: pd.DataFrame, subset: np.ndarray) -> tuple[float, list]:
    '''evaluates the accuracy of a given feautre subset'''
    # convert into column indexes
    ix = [i for i, x in enumerate(subset) if x]
    
    # check for slecection with no features
    if len(ix) == 0:
        return 0.0
    
    # select columns 
    X_new = X.iloc[:, ix] 
    
    # create a preprocessor
    preprocessor = create_preprocessor(X_new, subset)
    
    # create a pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(n_estimators=100))
    ])
    
    # evaluate model
    scores = cross_val_score(pipeline, X_new, y, scoring='accuracy', cv=3, n_jobs=1)
    
    # evaluate the meaen score
    result = np.mean(scores)
    
    return result, ix

def mutate(solution: np.ndarray, p_mutate: float) -> np.ndarry:
    '''makes a random alteration to a given feature subset, and returns the new subset'''
    # make a copy
    child = solution.copy()
    
    for i in range(len(child)):
        # check for a mutation
        if np.random.rand() < p_mutate:
            # flip the inclusion
            child[i] = not child[i]
    return child

def hillclimbing(X, y, objective, n_iter, p_mutate):
    '''
    starts with an inital feature subset
    for [n_iter]
        call mutate on the subset
        evaluate with objective
        replace the last subset with the new subset if it evaluated to a better metric
    returns the final subset
    '''
    # generate an initial point
    solution = np.random.choice([True, False], size=X.shape[1])    
    # evaluate the initial point
    solution_eval, ix = objective(X,y, solution)
    
    # run the hill climb
    for i in range(n_iter):
        # take a step
        candidate = mutate(solution, p_mutate)
        
        # evaluate candidate point
        candidate_eval, ix = objective(X, y, candidate)
        
        # check if we should keep the new point
        if candidate_eval >= solution_eval:
            # store the new point
            solution, solution_eval = candidate, candidate_eval
        
        # report progress
        print('>%d f(%s) = %f' % (i+1, len(ix), solution_eval))
    return solution, solution_eval