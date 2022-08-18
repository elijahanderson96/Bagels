import pandas as pd
from config.configs import *
import logging
import numpy as np

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Transform:
    def __init__(self, data):
        self.original_data = data
        self.labeled_data = None
        self.symbols = None
        self.report_dates = None

    @staticmethod
    def _impute_row_data(df):
        df.replace(0, np.nan, inplace=True)
        m = df.mean(axis=1)
        for i, col in enumerate(df):
            df.iloc[:, i] = df.iloc[:, i].fillna(m)
        return df

    def _interpolate_prep(self):
        """Sorts the values by symbols (if multiple) and report dates. Pops the report dates
        and symbols into a class attribute, and returns all numeric data from the original data"""
        self.original_data.sort_values(by=['symbol', 'reportdate'], inplace=True)
        self.original_data['reportdate'] = self.original_data['reportdate'].astype('str')
        self.original_data.drop_duplicates(inplace=True, subset=['reportdate'])
        self.report_dates = pd.to_datetime(self.original_data.pop('reportdate'))
        self.symbols = self.original_data.pop('symbol')
        return self.original_data._get_numeric_data()

    def _interpolate(self):
        """This function will take the original quarterly data and interpolate it such that
        there are data points for every day. It does this by injecting new data points for all days
        in between two quarterly reports in a linear fashion."""
        df = self._interpolate_prep()
        df = self._impute_row_data(df)
        assert len(self.symbols) == len(self.report_dates), 'Length of symbols and report dates does not match'
        for i in range(len(self.report_dates)):
            if i + 1 < len(self.report_dates):
                if self.symbols.iloc[i + 1] != self.symbols.iloc[i]:  # ensure stocks are the same
                    continue
                symbol = self.symbols.iloc[i]
                n_days = abs((self.report_dates.iloc[i + 1] - self.report_dates.iloc[i]).days)  # n days apart
                if n_days == 1: continue
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

    def _one_hot_encode(self, data):
        data = pd.get_dummies(data, columns=['symbol'])
        # data = data.join(dum_df)
        logger.info(f'The training dataset now has shape {data.shape} after one hot encoding')
        return data

    def create_feature_label_matrix(self):
        logger.info('Generating features and labels')
        logger.info(f'Interpolating data. Original data was of shape {self.original_data.shape}')
        data = self._interpolate()
        logger.info(f'Shape is now {data.shape} after interpolating')
        symbols = ', '.join(self.symbols.unique().tolist())
        if ',' in symbols: data = self._one_hot_encode(data)
        logger.info(f'Feature matrix created for {symbols}')
        labels = pd.read_sql('SELECT date, close '
                             "FROM market.stock_prices "
                             "WHERE symbol in %(symbols)s",
                             con=POSTGRES_URL,
                             params={'symbols': tuple(symbols.split(','))})
        logger.info('Fetching labels for aforementioned symbols')
        # one hot encode somewhere here
        data['date'] = pd.to_datetime(data['date']).dt.date
        labels['date'] = pd.to_datetime(labels['date']).dt.date
        self.labeled_data = pd.merge(left=data, right=labels, on=['date', 'symbol'])
        logger.info('Pasting labels on')
        return self.labeled_data


data = pd.read_sql("SELECT * FROM market.fundamentals WHERE symbol in ('A','BAC')", con=POSTGRES_URL)

self = FeaturePrep(data=data)
labeled_data = self.create_feature_label_matrix()
labeled_data.to_sql('model_data', con=POSTGRES_URL, schema='market', if_exists='replace')

# get max date of fundamentals table for each symbol and max date of interpolated table.
# if they dont match, find the entry that matches max and interpolate.
