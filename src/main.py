from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

import numpy as np
import pandas as pd

# imports from within the project
from preprocessor import create_preprocessor
from utils import clean_and_split
from report import create_report

# define training and validation dataframes 
train_X, valid_X, train_y, valid_y = clean_and_split(
    filename = "telco_data.csv",
    features = ["tenure", "MonthlyCharges", "Contract"],
    test_size = 0.3)

# setup preprocessor
preprocessor = create_preprocessor(train_X = train_X)

xgb_model = XGBClassifier(n_estimators=100)

# define the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb_model)
])

# fit data (and preprocess automatically with the pipeline)
pipeline.fit(train_X, train_y)

report = create_report(
    score_pairs = [("mae", 5.0)]
)

print("Run Successful!\n")
