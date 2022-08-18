import datetime

import pandas as pd
from config.configs import *
import logging
import numpy as np

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Transform:
    """Takes data from any endpoint and transform it. We impute row data such that
    any zeros are replaced by the average value of the row, and interpolate the data such
    that there's data for every day. It is a linear interpolation."""

    def __init__(self, data, table='fundamentals'):
        self.original_data = data
        self.labeled_data = None
        self.symbols = None
        self.report_dates = None
        self.table = table

    @staticmethod
    def _impute_row_data(df):
        df.replace(0, np.nan, inplace=True)
        m = df.mean(axis=1)
        for i, col in enumerate(df):
            df.iloc[:, i] = df.iloc[:, i].fillna(m)
        return df

    def _interpolate_prep(self):
        """Sorts the values by symbols (if multiple) and report dates. Pops the report dates (after adding 91 days)
        and symbols into a class attribute, and returns all numeric data from the original data"""
        self.original_data.sort_values(by=['symbol', 'reportdate'], inplace=True)
        self.original_data['reportdate'] = self.original_data['reportdate'].astype('str')
        self.original_data.drop_duplicates(inplace=True, subset=['reportdate'])
        self.report_dates = pd.to_datetime(self.original_data.pop('reportdate')) + datetime.timedelta(days=91)
        self.symbols = self.original_data.pop('symbol')
        return self.original_data._get_numeric_data()

    def _interpolate(self):
        """This function will take the original quarterly data and interpolate it such that
        there are data points for every day. It does this by injecting new data points for all days
        in between two quarterly reports in a linear fashion."""
        df = self._interpolate_prep()
        logger.info(f"Interpolating data for {', '.join(self.symbols.unique())}")
        df = self._impute_row_data(df)
        assert len(self.symbols) == len(self.report_dates), 'Length of symbols and report dates does not match'
        for i in range(len(self.report_dates)):
            if i + 1 < len(self.report_dates):
                if self.symbols.iloc[i + 1] != self.symbols.iloc[i]:  # ensure stocks are the same
                    continue
                symbol = self.symbols.iloc[i]
                n_days = abs((self.report_dates.iloc[i + 1] - self.report_dates.iloc[i]).days)  # n days apart
                if n_days == 1:
                    continue
                dates = pd.date_range((self.report_dates.iloc[i]),
                                      (self.report_dates.iloc[i + 1]), freq='d').strftime("%Y-%m-%d").tolist()
                temp_values = {}
                for col in df.columns:
                    if col != 'reportdate':
                        tmp = []
                        for j in range(n_days + 1):
                            # this linearizes the values
                            number = ((df[col].iloc[i + 1] - df[col].iloc[i]) / n_days * j) + df[col].iloc[i]
                            tmp.append(number)
                        temp_values.update({col: tmp})
                temp_values.update({'date': dates, 'symbol': symbol})
                tmp_df = pd.DataFrame(temp_values)
                df = pd.concat([df, tmp_df])
        return df  # .drop_duplicates(subset=['dates_interpolated', 'symbol'], inplace=True)

    def run(self):
        interpolated_data = self._interpolate()
        if len(self.symbols) > 1:
            interpolated_data.drop_duplicates(subset=['date', 'symbol'], inplace=True)
        logger.info('Inserting interpolated data into postgres')
        interpolated_data.to_sql(name=self.table+'_interpolated', con=POSTGRES_URL, schema='market', if_exists='append',index=False)
        logger.info('Data inserted!')


data = pd.read_sql("SELECT * FROM market.fundamentals WHERE symbol in ('AA')", con=POSTGRES_URL)
self = Transform(data=data)
self.run()

# get max date of fundamentals table for each symbol and max date of interpolated table.
# if they dont match, find the entry that matches max and interpolate.
