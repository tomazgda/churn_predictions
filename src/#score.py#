# score.py

import pandas as pd
import joblib
import logging

def main() -> None:
    '''Load pipeline, and scoring data -> produce some scores'''

    # load data to be scored
    scoring_data = pd.read_csv('data/scoring_data.csv', index_col=0)

    # load pipeline
    pipeline = joblib.load('pipelines/pipeline.joblib')

    # Generate scores
    scores = pipeline.predict(scoring_data)

    # write the scores as a DataFrame to a csv file
    pd.DataFrame( {"scores": scores}, index = scoring_data.index ).to_csv('data/scores.csv')

if __name__ == "__main__":
    main()
