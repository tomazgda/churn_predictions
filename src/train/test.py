# test.py

# NOTE : The purpose of this file is to evaluate other modules before calling them from main

from preprocessor import create_preprocessor
from features import objective, mutate, hillclimbing, filter_find_best_features

import numpy as np
import pandas as pd

# load data
df = pd.read_csv("data/telco.csv").dropna()

# split target from features
y = np.where(df['Churn'] == 'Yes', 1, 0)
X = df.drop(['Churn', 'TotalCharges', 'customerID'], axis = 1, inplace = False)

# begin feature selection
n_iter = 100
p_mut = 10.0 / 500.0
subset,score = hillclimbing(X, y, n_iter, p_mut)

# success message
print("Testing Successful!")

