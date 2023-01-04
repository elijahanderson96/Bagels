import pandas as pd
from config.configs import POSTGRES_URL

### This script is used to unpack model metadata and look at what scores are best.

predictions = pd.read_sql('SELECT * FROM market.models',con=POSTGRES_URL)

test_scores = predictions['test_scores'].to_dict()

print(test_scores[5]['AMZN']) #