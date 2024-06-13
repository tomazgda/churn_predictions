import numpy as np
import pandas as pd

# read csv into dataframe
data = pd.read_csv("telco_data.csv")

# write the first 10 rows of the dataframe into a new csv file
data.head(10).to_csv("telco_data_head.csv")

print("Run Successful!\n")
