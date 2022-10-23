import requests
import pandas as pd
import logging
from config.configs import *
from config.common import SYMBOLS
from datetime import date
from dateutil.relativedelta import relativedelta
from time import sleep

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Iex:
    def __init__(self):
        self.base_url = BASE_URL
        self.token = TOKEN
        self.engine = POSTGRES_URL
        self.version = 'v1/data/CORE/'

    @staticmethod
    def _json_to_dataframe(request):
        df = pd.DataFrame()
        for entry in request.json():
            temp = pd.DataFrame([entry], columns=list(entry.keys()))
            df = pd.concat([df, temp])
        return df

    @staticmethod
    def _shares_outstanding(symbol):
        """        GET / stock / {symbol} / stats / {stat?}"""
        url = f'https://cloud.iexapis.com/stable/stock/{symbol}/stats/sharesOutstanding'
        r = requests.get(url, params={'token': os.getenv('PRODUCTION_TOKEN')})
        return int(r.text)

    def _timeseries_inventory(self):
        url = 'https://cloud.iexapis.com/' + 'stable/' + 'time-series'
        r = requests.get(url=url, params={'token': os.getenv("PRODUCTION_TOKEN")})
        return self._json_to_dataframe(r)

    def _timeseries_metadata(self, time_series_id=None, stock=None, subkey='ttm'):
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


class Pipeline(Iex):
    def __init__(self):
        super().__init__()
        self.url = self.base_url + self.version

    # Fix the index of this dataframe
    def fundamental_valuations(self, stock, subkey='ttm', last=5):
        assert subkey in ['ttm', 'quarterly'], 'Subkey must be ttm or quarterly'
        logger.info(f'Grabbing latest {last} fundamentals reports for {stock}.')
        url = self.url + f'FUNDAMENTAL_VALUATIONS/{stock}/{subkey}'
        logger.info(f'Pinging {url} for fundamentals data')
        r = requests.get(url, params={'last': last, 'token': self.token})
        return self._json_to_dataframe(r)

    def cash_flow(self, stock, subkey='quarterly',last=1):
        logger.info(f'Grabbing latest {last} cash flow reports for {stock}')
        url = self.url + f'CASH_FLOW/{stock}/{subkey}'
        logger.info(f'Pinging {url} for cash flow data')
        r = requests.get(url, params={'last': last, 'token': self.token})
        return self._json_to_dataframe(r)

    def mortgage(self, last=1):
        logger.info(f'Grabbing last {last} mortgage reports')
        url = self.url + f'MORTGAGE'
        logger.info(f'Pinging {url} for mortgage data')
        r = requests.get(url, params={'last': last, 'token': self.token})
        return self._json_to_dataframe(r)

    def treasury_rates(self, last=5):
        """Only doing 10 year for now, id is dgs10"""
        logger.info(f'Grabbing last {last} treasury reports')
        url = self.url + f'TREASURY'
        logger.info(f'Pinging {url} for treasury data')
        r = requests.get(url, params={'token': self.token, 'last':last})
        return self._json_to_dataframe(r)

    def pull_latest(self):
        """Pull the latest data for a stock.
        Arguments: stock {str} stock to find number of available records for.
        """
        try:
            metadata = self._timeseries_metadata(time_series_id='CASH_FLOW', stock=stock)
            if metadata.empty:
                logger.warning(f'No data for {stock}')
                return
            try:
                print(metadata)
                n_records = int(metadata.loc[(metadata['subkey'] == 'QUARTERLY')]['count'])
            except:
                logger.warning('Could not find TTM in sub key.')
                n_records = 0
        except:
            n_records = 0
            logger.warning(f'Could not  source metadata for {stock}, skipping')

        try:
            current_records = int(pd.read_sql(f"SELECT COUNT(*) "
                                              f"FROM market.cash_flow "
                                              f"WHERE symbol='{stock}';",
                                              self.engine).squeeze())
        except:
            logger.info('Could not identify how many records were present.')
            y_n = input('Do you wish to pull all records? (y/n): ')
            if y_n == 'y':
                current_records = 0
            else:
                current_records = n_records

        n_records = n_records - current_records
        logger.info(f'Fetching {n_records} records from IEX cloud for {stock}')

        df = self.cash_flow(stock, last=n_records) if n_records > 0 else logger.info(
            f'Records up to date for {stock}')

        if df is not None and not df.empty:
            df.columns = map(str.lower, df.columns)  # make columns lowercase
            df.to_sql('cash_flow', self.engine, schema='market', if_exists='append', index=False)
        return df

    def run(self, stocks: list):
        for stock in stocks:
            self.pull_latest(stock)


if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.run(SYMBOLS)
