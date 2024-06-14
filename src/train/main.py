from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_absolute_error, accuracy_score

import joblib

import numpy as np
import pandas as pd

# imports from within the project
from preprocessor import create_preprocessor
from utils import clean_and_split
from report import create_report

def main() -> None:

    # define training and validation dataframes 
    train_X, valid_X, train_y, valid_y = clean_and_split(
        filename = "data/telco_data.csv",
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

    # save the pipeline
    joblib.dump(pipeline, 'pipelines/pipeline.joblib')

    # fit data (and preprocess automatically with the pipeline)
    pipeline.fit(train_X, train_y)

    # make some predictions
    predictions = pipeline.predict(valid_X)

    # generate a report
    report = create_report(
        score_pairs = [
            ("mae", mean_absolute_error(valid_y, predictions)),
            ("accuracy", accuracy_score(valid_y, predictions))
        ]
    )

    report.to_csv("report.csv")

    print("Run Successful!\n")

if __name__ == "__main__":
    main()