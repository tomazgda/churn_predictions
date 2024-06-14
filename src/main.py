import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

# import preprocessor defined in preprocessor.py
from preprocessor import Preprocessor

# import clean_and_split defined in utils.py
from utils import clean_and_split

# define training and validation dataframes 
features = ["tenure", "MonthlyCharges", "Contract"]
train_X, valid_X, train_y, valid_y = clean_and_split("telco_data.csv", features)

# setup preprocessor
preprocessor = Preprocessor(train_X)

xgb_model = XGBClassifier(n_estimators=100)

# define the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb_model)
])

# fit data (and preprocess automatically with the pipeline)
pipeline.fit(train_X, train_y)

print("Run Successful!\n")
