# Load pipeline, and hold_out data -> produce some scores

import pandas as pd
import joblib

def main() -> None:

    # load data to be scored
    scoring_data = pd.read_csv('data/scoring_data.csv')

    # load pipeline
    pipeline = joblib.load('pipelines/pipeline.joblib')

    # Generate scores
    scores = model.predict(scoring_data)
    
    # Log scores -- From Matt's Post https://towardsdatascience.com/deploy-a-lightgbm-ml-model-with-github-actions-781c094acfa3
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    f_handler = logging.FileHandler('data/scores.log')
    f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
    logger.info(f'Scores: {scores}')

if __name__ == "__main__":
    main()