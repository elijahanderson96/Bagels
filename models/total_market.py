import pandas as pd
import tensorflow as tf
import psycopg2
import logging
import os

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TotalMarketModel:
    def __init__(self):
        self.db_con = os.getenv('POSTGRES_CONNECTION')
        self.data = pd.read_sql('SELECT * FROM public.interpolated_fundamentals',con = self.db_con)
        self.labels = pd.read_sql('SELECT Close FROM public."Stock_Prices"',con = self.db_con)
        logger.info(f'Obtained dataset of shape {self.data.shape}')
        logger.info(f'Obtained labels of shape {self.labels}')


if __name__=='__main__':
    Model = TotalMarketModel()

