import requests
import pandas as pd
import numpy as np
import logging
from config.configs import *
from datetime import date
from dateutil.relativedelta import relativedelta


logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Iex:
    def __init__(self, production=False):
        if production:
            self.token = os.getenv('PRODUCTION_TOKEN')
            self.base_url = 'https://cloud.iexapis.com/'
            self.engine = os.getenv('POSTGRES_PROD_URL')
        else:
            self.token = os.getenv('SANDBOX_TOKEN')
            self.base_url = 'https://sandbox.iexapis.com/'
            self.engine = os.getenv('POSTGRES_DEV_URL')

        self.version = 'stable'

    @staticmethod
    def _json_to_dataframe(request):
        df = pd.DataFrame()
        for entry in request.json():
            temp = pd.DataFrame([entry], columns=list(entry.keys()))
            df = pd.concat([df, temp])
        return df

    @staticmethod
    def _impute_row_data(df):
        df.replace(0, np.nan, inplace=True)
        m = df.mean(axis=1)
        for i, col in enumerate(df):
            df.iloc[:, i] = df.iloc[:, i].fillna(m)
        return df


class Pipeline(Iex):
    def __init__(self, production=False):
        super().__init__(production)
        self.url = self.base_url + self.version + '/time-series/'
        self.active_endpoints = 'FUNDAMENTALS'

    def timeseries_inventory(self):
        url = 'https://cloud.iexapis.com/' + 'stable/' + 'time-series'
        r = requests.get(url=url, params={'token': os.getenv("PRODUCTION_TOKEN")})
        return self._json_to_dataframe(r)

    def timeseries_metadata(self, time_series_id=None, stock=None, subkey='ttm'):
        """See how many records IEX has to offer for a given time series for a given stock.
        Passing None for both time_series_id and stock gives you all the metadata
        (all time series for all stocks).
            Parameters:
                time_series_id: The time series endpoint.
                stock: the symbol in question.
                subkey: a filter parameter for the type of content returned.
            Returns:
                DataFrame"""
        assert subkey in ['ttm', 'quarterly', 'annual'], 'Subkey must be ttm,annual,or quarterly'
        url = f'https://cloud.iexapis.com/stable/metadata/time-series/'
        if time_series_id and stock and not subkey:
            url += f'{time_series_id}/{stock}'
        elif time_series_id and stock and subkey:
            url += f'{time_series_id}/{stock}'  # /{subkey}'
        elif not time_series_id and stock:
            url += f'*/{stock}'
        elif time_series_id and not stock:
            url += f'{time_series_id}'
        r = requests.get(url, params={'token': os.getenv("PRODUCTION_TOKEN")})
        return self._json_to_dataframe(r)

    # Fix the index of this dataframe
    def fundamentals(self, stock, subkey='ttm', last=5):
        assert subkey in ['ttm', 'quarterly'], 'Subkey must be ttm or quarterly'
        logger.info(f'Grabbing latest {last} fundamentals reports for {stock}.')
        url = self.url + f'/FUNDAMENTALS/{stock}/{subkey}'
        logger.info(f'Pinging {url} for fundamentals data')
        r = requests.get(url, params={'last': last, 'token': self.token})
        return self._json_to_dataframe(r)

    def treasury_rates(self, symbol, last=5):
        """Only doing 10 year for now, id is dgs10"""
        assert symbol in ['dgs10'], 'Only considering 10 year treasure rates at this time'
        last_n_years = date.today() - relativedelta(years=last)
        url = self.url + f'TREASURY/{symbol}'
        r = requests.get(url, params={'token': self.token, 'from': last_n_years})
        return self._json_to_dataframe(r)

    @staticmethod
    def shares_outstanding(symbol):
        """        GET / stock / {symbol} / stats / {stat?}"""
        url = f'https://cloud.iexapis.com/stable/stock/{symbol}/stats/sharesOutstanding'
        r = requests.get(url, params={'token': os.getenv('PRODUCTION_TOKEN')})
        return int(r.text)

    def pull_latest(self, stock):
        """Pull the latest data for a stock.
        Arguments: stock {str} stock to find number of available records for.
        """
        # Probe IEX Cloud to see how many records they have
        # when we have multiple endpoints we should incorporate some dictionaries to
        # pull latest
        metadata = {}  # -> perhaps a dict comprehension to retrieve metadata with form
                        # {'FUNDAMENTALS': metadata_df, 'TREASURY': metadata_df}
        metadata = self.timeseries_metadata(time_series_id='FUNDAMENTALS', stock=stock)
        # Retrieve the "count" entry in the metadata
        n_records = metadata.loc[(metadata['subkey'] == 'TTM')]['count']
        logger.info(f'There were {int(n_records)} TTM records found on IEX Cloud.')
        # TODO Why is this not count?
        #try: current_records = pd.read_sql(f"SELECT * FROM fundamentals.fundamentals WHERE symbol='{stock}';", self.engine)
        #current_records = len(current_records.loc[(current_records['symbol'] == stock)]['symbol'])
        current_records = 0
        n_records = int(n_records) - int(current_records)
        df = self.fundamentals(stock, last=n_records) if n_records != 0 else logger.info(
            f'Records up to date for {stock}')
        if not df.empty: df.to_sql('fundamentals', self.engine, schema='fundamentals', if_exists='replace',index=False)
        # Return the number of records we need to fetch, so that the int is accessible via XCOM (Cross communication,
        # a way to pass parameters between tasks in a DAG)

    def run(self, stocks: list):
        for stock in stocks:
            self.pull_latest(stock)
            shares_outstanding = self.shares_outstanding(stock)
            prices = UpdatePrices(stock, shares_outstanding)


if __name__ == '__main__':
    DataGetter = Pipeline()
    DataGetter.run(['KO', 'MSFT'])
