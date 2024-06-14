import pandas as pd

def create_report(score_pairs: list[tuple(str, float | int)]) -> pd.DataFrame:
    """run some metrics and list them in a dataframe"""

    report = pd.DataFrame(
        data = {"score" : [score_pair[1] for score_pair in score_pairs]}, 
        index = [score_pair[0] for score_pair in score_pairs])

    return report