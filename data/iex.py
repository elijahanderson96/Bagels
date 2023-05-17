import logging
from datetime import datetime
from traceback import print_exc

import pandas as pd
import requests

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
        self.url = self.base_url + self.version

    @staticmethod
    def json_to_dataframe(request):
        tmp = []
        for entry in request.json():
            temp = pd.DataFrame([entry], columns=list(entry.keys()))
            tmp.append(temp)
        df = pd.concat(tmp) if tmp else pd.DataFrame()
        return df

    @staticmethod
    def shares_outstanding(symbol):
        """        GET / stock / {symbol} / stats / {stat?}"""
        url = f'https://cloud.iexapis.com/stable/stock/{symbol}/stats/sharesOutstanding'
        r = requests.get(url, params={'token': os.getenv('PRODUCTION_TOKEN')})
        if r.text == 'Unknown symbol':
            return 0
        return int(r.text)


class Pipeline(Iex):
    def __init__(self):
        super().__init__()

    def fundamental_valuations(self, from_date, to_date=datetime.today().strftime('%Y-%m-%d')):
        logger.info(f'Grabbing all fundamentals reports data from {from_date}.')
        url = self.url + f'FUNDAMENTAL_VALUATIONS'
        logger.info(f'Pinging {url} for fundamentals data')
        r = requests.get(url, params={'from': from_date, 'to': to_date, 'token': self.token})
        return self.json_to_dataframe(r)

    def cash_flow(self, from_date, to_date=datetime.today().strftime('%Y-%m-%d')):
        logger.info(f'Grabbing data from {from_date} regarding cash flow reports.')
        url = self.url + f'CASH_FLOW'
        logger.info(f'Pinging {url} for cash flow data')
        r = requests.get(url, params={'from': from_date, 'to': to_date, 'token': self.token})
        return self.json_to_dataframe(r)

    def mortgage(self, from_date, to_date=datetime.today().strftime('%Y-%m-%d')):
        logger.info(f'Grabbing data from {from_date} for mortgage reports.')
        url = self.url + f'MORTGAGE'
        logger.info(f'Pinging {url} for mortgage data')
        r = requests.get(url, params={'from': from_date, 'to': to_date, 'token': self.token})
        df = self.json_to_dataframe(r)
        return df[['date', 'value', 'key']].drop_duplicates()

    def treasury_rates(self, from_date, to_date=datetime.today().strftime('%Y-%m-%d')):
        """Only doing 10 year for now, id is dgs10"""
        logger.info(f'Grabbing data from {from_date} for treasury reports.')
        url = self.url + f'TREASURY'
        logger.info(f'Pinging {url} for treasury data')
        r = requests.get(url, params={'from': from_date, 'to': to_date, 'token': self.token})
        df = self.json_to_dataframe(r)
        return df[['date', 'value', 'key']].drop_duplicates()

    def economic(self, from_date, to_date=datetime.today().strftime('%Y-%m-%d')):
        logger.info(f'Grabbing data from {from_date} for treasury reports.')
        url = self.url + f'ECONOMIC'
        logger.info(f'Pinging {url} for economic data')
        r = requests.get(url, params={'from': from_date, 'to': to_date, 'token': self.token})
        df = self.json_to_dataframe(r)
        return df[['date', 'value', 'key']].drop_duplicates()

    def fx_rates(self, from_date, to_date=datetime.today().strftime('%Y-%m-%d')):
        logger.info(f'Grabbing data from {from_date} for foreign exchange datapoints.')
        url = self.url + f'FX-DAILY'
        logger.info(f'Pinging {url} for foreign exchange data')
        r = requests.get(url, params={'from': from_date, 'to': to_date, 'token': self.token})
        return self.json_to_dataframe(r)

    def energy(self, from_date, to_date=datetime.today().strftime('%Y-%m-%d')):
        logger.info(f'Grabbing data from {from_date} for energy reports.')
        url = self.url + f'ENERGY'
        logger.info(f'Pinging {url} for energy data')
        r = requests.get(url, params={'from': from_date, 'to': to_date, 'token': self.token})
        df = self.json_to_dataframe(r)
        return df[['date', 'value', 'key']].drop_duplicates()

    def commodities(self):
        pass

    def ping_endpoint(self, endpoint_name, pull_from, to_date):
        """This is a rather crude but effective way of implementing which
        function to call when we invoke the update_data method."""
        df = pd.DataFrame()

        if endpoint_name == 'fundamental_valuations':
            df = self.fundamental_valuations(pull_from, to_date)

        if endpoint_name == 'cash_flow':
            df = self.cash_flow(pull_from, to_date)

        if endpoint_name == 'treasury':
            df = self.treasury_rates(pull_from, to_date)

        if endpoint_name == 'mortgage':
            df = self.mortgage(pull_from, to_date)

        if endpoint_name == 'economic':
            df = self.economic(pull_from, to_date)

        if endpoint_name == 'energy':
            df = self.energy(pull_from, to_date)

        if endpoint_name == 'historical_prices':
            tmp = []
            for symbol in ('MS', 'GS', 'JPM'):
                tmp.append(self.historical_stock_prices(symbol, from_date='2000-01-01'))
            df = pd.concat(tmp)

        try:

            if df.empty:
                return logger.info(f'No data from {pull_from} to {to_date}')

            df.to_sql(endpoint_name.lower(), con=POSTGRES_URL, index=False, if_exists='append', schema='bagels')

        except Exception:
            print_exc()
            logger.warning(f'Could not insert data for {endpoint_name}.')

        return logger.info(f'{endpoint_name} updated with records from {pull_from}')

    def update_data(self, from_scratch=False):
        """Pull the latest data for a stock.
        Arguments: stock {str} stock to find number of available records for.
        """
        endpoint_metadata = pd.read_sql('SELECT * FROM bagels.pipeline_metadata', con=POSTGRES_URL)
        days = 100 if from_scratch else 1
        for from_date, endpoint in endpoint_metadata.itertuples(index=False, name=None):
            # there is a huge flaw in that they can only return 5000 data points at a time. We
            # will attempt to iterate by month to back fill data.
            logger.info(f'Updating {endpoint}, pulling all data from {from_date} to present.')
            try:
                while pd.to_datetime(from_date) < datetime.today():
                    to_date = pd.to_datetime(from_date) + pd.DateOffset(days=days)
                    self.ping_endpoint(endpoint, from_date, to_date)
                    from_date = pd.to_datetime(from_date) + pd.DateOffset(days=days)
            except:
                print_exc()

        logger.info('All data has been updated')
