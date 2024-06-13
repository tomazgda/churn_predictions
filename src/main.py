import numpy as np
import pandas as pd

# import preprocessor defined in preprocessor.py
from preprocessor import *

# read csv into dataframe
data = pd.read_csv("telco_data.csv")

# write the first 10 rows of the dataframe into a new csv file
data.head(10).to_csv("telco_data_head.csv")

# remove rows with missing targets
data_without_na = data.dropna(axis=0, subset=['Churn'], inplace=False)

# seperate target from features, and encode target data
y = np.where(data_without_na['Churn'] == 'Yes', 1, 0)

# drop target from whole data set, return features which will be called X
X = data_without_na.drop(['Churn'], axis=1, inplace=False)[["tenure", "MonthlyCharges", "Contract"]]

# seperature training from validation data
from sklearn.model_selection import train_test_split 

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=0)

# series descripting of columns are of dtype object
L = (train_X.dtypes == 'object')

# Obtain a list of all columns containing categorical variables in the training data
categorical_cols = list(L[L].index)

# Create a list of all columns containg numerical variables in the training data 
numerical_cols = [e for e in L.index if not (e in L[L].index)]

# setup preprocessor
preprocessor = Preprocessor(numerical_cols, categorical_cols)

from xgboost import XBGClassifier
xgb_model = XGBClassifier(n_estimators=100)

# define the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb_model)
])

# fit data (and preprocess automatically with the pipeline)
pipeline.fit(train_X, train_y)

print("Run Successful!\n")
