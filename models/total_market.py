import pandas as pd
import tensorflow as tf
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TotalMarketModel:
    def __init__(self):
        self.db_con = os.getenv('POSTGRES_CONNECTION')
        self.data = pd.read_sql('SELECT * FROM public.interpolated_fundamentals',con=self.db_con)
        self.data.rename(columns={'dates_interpolated':'Date'},inplace=True)
        self.data['Date']=self.data['Date'].astype('datetime64[ns]')
        self.data['Date'] = self.data['Date'] + pd.Timedelta(days=91)
        self.labels = pd.read_sql('SELECT * FROM public."Stock_Prices"',con=self.db_con)
        self.labels['Date'] = self.labels['Date'].astype('datetime64[ns]')
        logger.info(f'Obtained dataset of shape {self.data.shape}')
        logger.info(f'Obtained labels of shape {self.labels}')
        self.dataset = pd.merge(self.data, self.labels, on=['Date','symbol'])
        logger.info(f'Dataset is of shape {self.dataset.shape}')
        print(self.dataset.head(5))


if __name__=='__main__':
    Model = TotalMarketModel()

