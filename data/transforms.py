import datetime
import logging

import pandas as pd

from config.common import QUERIES
from config.configs import POSTGRES_URL

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class FeaturePrepper:
    def __init__(self, datasets: list, params: dict):
        self.datasets = datasets
        self.params = params
        self.data = None

    def _fetch_raw_data(self):
        """Fetch the raw data from db and map it."""
        #  fetch the un-parameterized queries from the dict of sql queries
        queries = {q: QUERIES[q] for q in self.datasets}

        #  parameterize the queries (all we do is string substitution)
        for placeholder, parameter in self.params.items():
            queries = {k: q.replace(placeholder, parameter) for k, q in queries.items()}

        #  return appropriate data
        return {q_name: pd.read_sql(q_value, con=POSTGRES_URL) for q_name, q_value in queries.items()}

    def _join_datasets(self, left='fundamental_valuations', max_date='right'):
        """Join datasets based on date. Ensure that in our sql queries, every date column we're joining on
        is aliased as "SELECT date_column as date" otherwise this will fail.

        Args:
            left: The dataframe that will start the join as the left most df.
             Pass a string indicating which dataset should be the root join df.

        Returns:
            data (df): dataframe containing all features joined together by date.

        """
        logger.info(f'Forming full dataset with all features. The {left} dataframe will '
                    f'be used as a starting point, all other dataframes will be joined to it. '
                    f'The feature matrix will contain the following {", ".join(self.datasets)}')

        self.raw_data = self._fetch_raw_data()
        left_join_df = self.raw_data[left]
        logger.info(f'Starting data is of shape {left_join_df.shape}.')

        for dataset, contents in self.raw_data.items():
            if dataset != left:
                contents = self._transform(contents)
                if max(contents['date']) <= datetime.datetime.now():
                        logger.warning(f'{dataset} is not up to date. Consider re-ingesting the data. '
                                       f'We cannot build models with this feature as it stands.')
                else:
                    logger.info(f'{dataset}')
                    left_join_df = left_join_df.merge(contents, on='date').drop_duplicates()
                    logger.info(f'{left_join_df.shape}')

        logger.info(f'Final data is of shape {left_join_df.shape}.')
        print(left_join_df.sort_values(by='date', ascending=False)[['symbol', 'date']].head(25))
        return left_join_df.drop_duplicates(subset=['symbol', 'date'])

    @staticmethod
    def _interpolate(df_chunk):
        n_days = abs((df_chunk['date'].iloc[0] - df_chunk['date'].iloc[1]).days)

        if n_days == 1:
            # date is already 1 day apart for this chunk, so we don't need to do anything.
            return df_chunk

        dates = pd.date_range(df_chunk['date'].iloc[0], df_chunk['date'].iloc[1], freq='d')
        temp_values = {}

        for col in df_chunk.columns:
            if col != 'date':
                tmp = []
                for j in range(n_days + 1):
                    number = ((df_chunk[col].iloc[1] - df_chunk[col].iloc[0]) / n_days * j) + df_chunk[col].iloc[0]
                    tmp.append(number)
                temp_values.update({col: tmp})
        temp_values['date'] = dates
        return pd.DataFrame(temp_values)

    def _transform(self, df: pd.DataFrame):
        """Sometimes we can't join data based on date because it might be quarterly, weekly, monthly, etc.
        This function will linearize all data between two given dates as too approximate values on
        a daily basis rather than quarterly, weekly, monthly, etc."""
        df['date'] = df['date'].astype('datetime64[ns]')
        df.drop_duplicates(subset=['date'], inplace=True)
        df.sort_values(inplace=True, by='date')
        chunked_data = [df.iloc[i:i + 2] for i in range(len(df) - 1)]
        df_interpolated = pd.concat([self._interpolate(chunk) for chunk in chunked_data], ignore_index=True)
        return df_interpolated

    def create_feature_matrix(self):
        self.data = self._join_datasets()
        return self.data


if __name__ == '__main__':
    self = FeaturePrepper(['fundamental_valuations',
                                         'fetch_5Ymortgage_rates',
                                         'fetch_15Ymortgage_rates',
                                         'fetch_30Ymortgage_rates',
                                         'fetch_recession_probability',
                                         'fetch_num_total_employees',
                                         'fetch_housing_starts',
                                         'fetch_industrial_production',
                                         'fetch_unemployment_rate',
                                         'fetch_vehicle_sales',
                                         'fetch_cpi',
                                         'fetch_unemployment_claims',
                                         'fetch_comm_paper_outstanding',
                                         'fetch_fed_funds',
                                         'fetch_real_gdp',
                                         'fetch_crude_oil_brent',
                                         'fetch_henry_hub_natural_gas',
                                         'fetch_jet_fuel',
                                         'fetch_regular_conventional_gas',
                                         'fetch_midgrade_conventional_gas',
                                         'fetch_diesel',
                                         'fetch_gas_russia',
                                         'fetch_heating_oil',
                                         'fetch_crude_oil_wti',
                                         'fetch_propane'],
                          params={'SYMBOLS': str(('JPM', 'MS', 'GS'))})

    left = 'fundamental_valuations'
