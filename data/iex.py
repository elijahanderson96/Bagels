import logging
import pandas as pd
import requests
from datetime import date
from dateutil.relativedelta import relativedelta
from sqlalchemy import create_engine
from time import sleep
from traceback import print_exc

from config.common import SYMBOLS
from config.configs import *

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

logging.getLogger("urllib3").setLevel(logging.WARNING)


class Iex:
    def __init__(self):
        self.base_url = BASE_URL
        self.token = TOKEN
        self.engine = POSTGRES_URL
        self.version = 'v1/data/CORE/'
        self.stock_endpoints = ['FUNDAMENTAL_VALUATIONS']
        self.market_endpoints = ['MORTGAGE', 'TREASURY']
        self.endpoints = self.market_endpoints + self.stock_endpoints
        self.current_tables = pd.read_sql("SELECT table_name "
                                          "FROM information_schema.tables "
                                          "WHERE table_schema = 'market';",
                                          POSTGRES_URL)['table_name'].to_list()

    @staticmethod
    def json_to_dataframe(request):
        df = pd.DataFrame()
        for entry in request.json():
            temp = pd.DataFrame([entry], columns=list(entry.keys()))
            df = pd.concat([df, temp])
        return df

    @staticmethod
    def shares_outstanding(symbol):
        """        GET / stock / {symbol} / stats / {stat?}"""
        url = f'https://cloud.iexapis.com/stable/stock/{symbol}/stats/sharesOutstanding'
        r = requests.get(url, params={'token': os.getenv('PRODUCTION_TOKEN')})
        if r.text == 'Unknown symbol': return 0
        return int(r.text)

    def timeseries_inventory(self):
        url = 'https://cloud.iexapis.com/' + 'stable/' + 'time-series'
        r = requests.get(url=url, params={'token': os.getenv("PRODUCTION_TOKEN")})
        return self.json_to_dataframe(r)

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
        return self.json_to_dataframe(r)

    def examine_current_records(self, table: str, symbol: str) -> int:
        """This function looks at what tables exist in postgres and then simply returns the count
        of records for a symbol in that table. If a table doesn't exist, we return a count of 0"""
        table = table.lower()
        if table not in self.current_tables:
            logger.warning(f'{table} DOES NOT EXIST IN DATABASE. PULLING ALL RECORDS')
            return 0  # indicates we have 0 records since the table does not exist
        engine = create_engine(POSTGRES_URL)
        current_records = {symbol: pd.read_sql(f'SELECT count(*) '
                                               f'FROM market.{table} '
                                               f"WHERE key='{symbol}'", engine).squeeze()}
        engine.dispose()

        return current_records[symbol]


class Pipeline(Iex):
    def __init__(self):
        super().__init__()
        self.url = self.base_url + self.version

    # Fix the index of this dataframe
    def fundamental_valuations(self, stock, last=1):
        logger.info(f'Grabbing latest {last} fundamentals reports for {stock}.')
        url = self.url + f'FUNDAMENTAL_VALUATIONS/{stock}'
        logger.info(f'Pinging {url} for fundamentals data')
        r = requests.get(url, params={'last': last, 'token': self.token})
        return self.json_to_dataframe(r)

    def cash_flow(self, stock, last=1):
        logger.info(f'Grabbing latest {last} cash flow reports for {stock}')
        url = self.url + f'CASH_FLOW/{stock}'
        logger.info(f'Pinging {url} for cash flow data')
        r = requests.get(url, params={'last': last, 'token': self.token})
        return self.json_to_dataframe(r)

    def mortgage(self, term, last=1):
        logger.info(f'Grabbing last {last} mortgage reports for {term}')
        url = self.url + f'MORTGAGE/{term}'
        logger.info(f'Pinging {url} for mortgage data')
        r = requests.get(url, params={'last': last, 'token': self.token})
        return self.json_to_dataframe(r)

    def treasury_rates(self, term, last=1, ):
        """Only doing 10 year for now, id is dgs10"""
        logger.info(f'Grabbing last {last} treasury reports for {term}')
        url = self.url + f'TREASURY/{term}'
        logger.info(f'Pinging {url} for treasury data')
        r = requests.get(url, params={'token': self.token, 'last': last})
        return self.json_to_dataframe(r)

    def ping_endpoint(self, endpoint_name, symbol, records_to_pull=0):
        """This is a rather crude but effective way of implementing which
        function to call when we invoke the update_data method."""
        df = pd.DataFrame()

        if records_to_pull == 0: return logger.info(f'Records for {symbol} within {endpoint_name} are up to date.')
        if endpoint_name == 'FUNDAMENTAL_VALUATIONS':
            df = self.fundamental_valuations(symbol, records_to_pull)
        if endpoint_name == 'CASH_FLOW':
            df = self.cash_flow(symbol, records_to_pull)
        if endpoint_name == 'TREASURY':
            df = self.treasury_rates(symbol, records_to_pull)
        if endpoint_name == 'MORTGAGE':
            df = self.mortgage(symbol, records_to_pull)

        try:
            df.to_sql(endpoint_name.lower(), con=POSTGRES_URL, index=False, if_exists='append', schema='market')
        except Exception:
            print_exc()
            logger.warning(f'Could not insert data for {symbol} within {endpoint_name}.')

        return logger.info(f'{symbol} within {endpoint_name} updated with {records_to_pull} records')

    def update_data(self):
        """Pull the latest data for a stock.
        Arguments: stock {str} stock to find number of available records for.
        """
        endpoint_metadata = [{endpoint: self.timeseries_metadata(endpoint)} for endpoint in self.endpoints]
        for endpoint in endpoint_metadata:
            for endpoint_name, endpoint_info in endpoint.items():
                keys_and_counts = dict(zip(endpoint_info['key'], endpoint_info['count']))
                for key, count in keys_and_counts.items():
                    sleep(.1)
                    current_records = self.examine_current_records(endpoint_name, key)
                    records_to_pull = count - current_records
                    try:
                        self.ping_endpoint(endpoint_name, key, records_to_pull)
                    except:
                        print_exc()

        logger.info('All data has been updated')


if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.update_data()
