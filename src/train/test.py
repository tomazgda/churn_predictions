# test.py

# NOTE : The purpose of this file is to evaluate other modules before calling them from main

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from preprocessor import create_preprocessor
from features import objective, mutate, hillclimbing, filter_find_best_features

from preprocessor import create_preprocessor
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score

import numpy as np
import pandas as pd

# load data
df = pd.read_csv("data/telco_data.csv").dropna()

# split target from features
y = np.where(df['Churn'] == 'Yes', 1, 0)
X = df.drop(['Churn', 'TotalCharges', 'customerID'], axis = 1, inplace = False)

# define model
model = RandomForestClassifier(n_estimators=100)

# define metric for evalution   
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# begin feature selection
n_iter = 1 # TODO Increase number
p_mut = 10.0 / 100.0
subset,score = hillclimbing(X, y, n_iter, p_mut, model)

# return the best feature set -> TODO use that to build the final model

# convert into column indexes
ix = [i for i, x in enumerate(subset) if x]
print('Done!')
print('Best: f(%d) = %f' % (len(ix), score))

# fit a new pipeline with the subset

# create a preprocessor
preprocessor = create_preprocessor(X.iloc[:, ix])
    
# create a pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# success message
print("Testing Successful!")
