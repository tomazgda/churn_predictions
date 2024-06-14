# main.py

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split

import joblib

import numpy as np
import pandas as pd

# imports from within the project
from preprocessor import create_preprocessor
from utils import clean_and_split
from report import create_report

def main() -> None:

    # read csv into a DataFrame
    dataset = pd.read_csv("data/telco_data.csv", index_col=0)

    # Split off data for scores later
    scoring_data, train_test = train_test_split(dataset, test_size = 0.1, random_state=0)

    # save holdout data to file: has all the features
    scoring_data.drop(['Churn'], axis=1, inplace=False).to_csv("data/scoring_data.csv")

    # define training and testing dataframes 
    train_X, test_X, train_y, test_y = clean_and_split(
        data = train_test,
        features = ["tenure", "MonthlyCharges", "Contract"],
        test_size = 0.3)

    # setup preprocessor
    preprocessor = create_preprocessor(train_X = train_X)

    # create the model
    xgb_model = XGBClassifier(n_estimators=100)

    # define the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', xgb_model)
    ])

    # fit data (and preprocess automatically with the pipeline)
    pipeline.fit(train_X, train_y)

    # save the pipeline
    joblib.dump(pipeline, 'pipelines/pipeline.joblib')

    # make some predictions
    predictions = pipeline.predict(test_X)

    # generate a report
    report = create_report(
        score_pairs = [
            ("mae", mean_absolute_error(test_y, predictions)),
            ("accuracy", accuracy_score(test_y, predictions))
        ]
    )

    # save the report to file
    report.to_csv("data/report.csv")

    print("Run Successful!\n")

if __name__ == "__main__":
    main()