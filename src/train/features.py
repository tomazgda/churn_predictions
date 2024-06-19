# features.py

# NOTE : None of the code in this file has been tested

# TODO : pass model into feature selection functions

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from itertools import product

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import pandas as pd

from preprocessor import create_preprocessor

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


# note sure of the type of model 
def objective(X: pd.DataFrame, y: pd.DataFrame, subset: np.ndarray, model) -> tuple[float, list]:
    '''evaluates the accuracy of a given feautre subset'''
    # convert into column indexes
    ix = [i for i, x in enumerate(subset) if x]
    
    # check for slecection with no features
    if len(ix) == 0:
        return 0.0
    
    # select columns 
    X_new = X.iloc[:, ix] 
    
    # create a preprocessor
    preprocessor = create_preprocessor(X_new)
    
    # create a pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # evaluate model
    scores = cross_val_score(pipeline, X_new, y, scoring='accuracy', cv=3, n_jobs=1)
    
    # evaluate the meaen score
    result = np.mean(scores)
    
    return result, ix

def mutate(solution: np.ndarray, p_mutate: float) -> np.ndarray:
    '''makes a random alteration to a given feature subset, and returns the new subset'''
    # make a copy
    child = solution.copy()
    
    for i in range(len(child)):
        # check for a mutation
        if np.random.rand() < p_mutate:
            # flip the inclusion
            child[i] = not child[i]
    return child

def hillclimbing(X: pd.DataFrame, y: pd.DataFrame, n_iter: float, p_mutate: float, model) -> tuple[np.ndarray, float]:
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
    solution_eval, ix = objective(X,y, solution, model)
    
    # run the hill climb
    for i in range(n_iter):
        # take a step
        candidate = mutate(solution, p_mutate)
        
        # evaluate candidate point
        candidate_eval, ix = objective(X, y, candidate, model)
        
        # check if we should keep the new point
        if candidate_eval >= solution_eval:
            # store the new point
            solution, solution_eval = candidate, candidate_eval
        
        # report progress
        print('>%d f(%s) = %f' % (i+1, len(ix), solution_eval))
    return solution, solution_eval

# filter feature selection uses statistical techniques to evaluate the relationship between each feature and the target
def filter_find_best_features():
    '''Uses filter feature selection to find the feature selection with the highest accuracy'''
    pass