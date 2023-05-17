import logging
from datetime import date

import pandas as pd
from data.iex import Pipeline
from data.labels import Prices
from config.configs import POSTGRES_URL
from models.models import PredictionPipeline

today = date.today()

# Month abbreviation, day and year
today = today.strftime("%b_%d_%Y")
logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger('run.py')
logging.getLogger('h5py._conv').setLevel(logging.DEBUG)

pd.set_option('display.max_rows', 500)

if __name__ == '__main__':
    DataGetter = Pipeline()
    #DataGetter.update_data(from_scratch=False)
    #prices = Prices()
    #prices.update_db()

    symbols = pd.read_sql("SELECT DISTINCT(symbol) FROM bagels.labels;",con=POSTGRES_URL)['symbol'].to_list()
    model = PredictionPipeline(symbols=['JPM','GS'], validate=False,model_type='classify', features=['fundamental_valuations'])
    model.train()
