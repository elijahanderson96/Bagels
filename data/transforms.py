import datetime
from multiprocessing import Pool, cpu_count
import pandas as pd
from config.configs import *
import logging
import numpy as np

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class FeaturePrepper:
    """Takes data from any endpoint and transform it. We impute row data such that
    any zeros are replaced by the average value of the row, and interpolate the data such
    that there's data for every day. It is a linear interpolation."""

    def __init__(self, data):
        self.original_data = data
        self.chunked_data = None
        self.labeled_data = None
        self.symbols = None
        self.report_dates = None
        self.preprocess()

    @staticmethod
    def interpolate(df_chunk):
        assert df_chunk['symbol'].iloc[0] == df_chunk['symbol'].iloc[1], 'symbols do not match'
        df_chunk['reportdate'] = df_chunk['reportdate'].astype('datetime64')
        n_days = abs((df_chunk['reportdate'].iloc[0] - df_chunk['reportdate'].iloc[1]).days)
        dates = pd.date_range(df_chunk['reportdate'].iloc[0], df_chunk['reportdate'].iloc[1], freq='d')
        temp_values = {}
        for col in df_chunk.columns:
            if col not in ('reportdate', 'symbol'):
                tmp = []
                for j in range(n_days + 1):
                    number = ((df_chunk[col].iloc[1] - df_chunk[col].iloc[0]) / n_days * j) + df_chunk[col].iloc[0]
                    tmp.append(number)
                temp_values.update({col: tmp})
        temp_values.update({'date': dates, 'symbol': df_chunk['symbol'].iloc[0]})
        return pd.DataFrame(temp_values)

    def preprocess(self):
        """Sorts the values by symbols (if multiple) and report dates. Pops the report dates (after adding 91 days)
        and symbols into a class attribute, and returns all numeric data from the original data"""
        self.original_data.sort_values(by=['symbol', 'reportdate'], inplace=True)
        self.original_data.drop_duplicates(inplace=True, subset=['symbol', 'reportdate'])  # add symbol
        self.original_data['reportdate'] = pd.to_datetime(self.original_data['reportdate']) + datetime.timedelta(
            days=91)
        report_dates = self.original_data['reportdate'].to_list()
        symbols = self.original_data['symbol'].to_list()
        self.original_data = self.original_data._get_numeric_data()
        self.original_data['reportdate'] = report_dates
        self.original_data['symbol'] = symbols
        # keep only the columns in df that do not contain string
        self.chunked_data = [self.original_data.iloc[i:i + 2] for i in range(len(self.original_data) - 1)
                             if self.original_data['symbol'].iloc[i] == self.original_data['symbol'].iloc[i + 1]]

    @staticmethod
    def _impute_row_data(df):
        df.replace(0, np.nan, inplace=True)
        m = df.mean(axis=1)
        for i, col in enumerate(df):
            df.iloc[:, i] = df.iloc[:, i].fillna(m)
        return df

    def transform(self):
        logger.info(f'Provisioning {cpu_count() - 1} cpus for transformation')
        with Pool(cpu_count() - 1) as p:
            dfs = p.map(self.interpolate, self.chunked_data)
            df = pd.concat(dfs, ignore_index=True)
        df = self._impute_row_data(df)
        logger.info(f'Interpolation for {df["symbol"].unique()} complete')
        return df


if __name__ == '__main__':
    data = pd.read_sql("SELECT * FROM market.fundamentals", con=POSTGRES_URL)
    self = FeaturePrepper(data=data)
    data = self.transform()


