import logging
from datetime import datetime
from time import sleep
from traceback import print_exc

import pandas as pd
import requests
from sqlalchemy import create_engine
from dateutil.relativedelta import relativedelta


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
        self.market_endpoints = ['MORTGAGE', 'ECONOMIC', 'ENERGY', 'FX-DAILY']
        self.endpoints = self.market_endpoints + self.stock_endpoints
        self.current_tables = pd.read_sql("SELECT table_name "
                                          "FROM information_schema.tables "
                                          "WHERE table_schema = 'market';",
                                          POSTGRES_URL)['table_name'].to_list()
        self.tables = [table.lower() for table in (self.stock_endpoints + self.market_endpoints)]

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
        if r.text == 'Unknown symbol':
            return 0
        return int(r.text)


class Pipeline(Iex):
    def __init__(self):
        super().__init__()
        self.url = self.base_url + self.version

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
        return self.json_to_dataframe(r)

    def treasury_rates(self, from_date, to_date=datetime.today().strftime('%Y-%m-%d')):
        """Only doing 10 year for now, id is dgs10"""
        logger.info(f'Grabbing data from {from_date} for treasury reports.')
        url = self.url + f'TREASURY'
        logger.info(f'Pinging {url} for treasury data')
        r = requests.get(url, params={'from': from_date, 'to': to_date, 'token': self.token})
        return self.json_to_dataframe(r)

    def economic(self, from_date, to_date=datetime.today().strftime('%Y-%m-%d')):
        logger.info(f'Grabbing data from {from_date} for treasury reports.')
        url = self.url + f'ECONOMIC'
        logger.info(f'Pinging {url} for economic data')
        r = requests.get(url, params={'from': from_date, 'to': to_date, 'token': self.token})
        return self.json_to_dataframe(r)

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
        return self.json_to_dataframe(r)

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
            df['date'] = df['date'].apply(
                lambda row: datetime.fromtimestamp(int(row) / 1000).strftime('%Y-%m-%d %H:%M:%S'))
        if endpoint_name == 'economic':
            df = self.economic(pull_from, to_date)
        if endpoint_name == 'energy':
            df = self.energy(pull_from, to_date)

        #if endpoint_name == 'fx-daily':
        #    df = self.fx_rates(pull_from, to_date)
        #    df['date'] = df['date'].apply(
        #        lambda row: datetime.fromtimestamp(int(row) / 1000).strftime('%Y-%m-%d %H:%M:%S'))

        try:

            if df.empty:
                return logger.info(f'No data from {pull_from} to {to_date}')

            # they unfortunately may have epoch time in these datasets that we have to cast.
            if endpoint_name in ('economic', 'energy'):
                epoch_time = df.loc[df['frequency'].isna()] if 'frequency' in df.columns.to_list() else None
                if epoch_time:
                    epoch_time['date'] = epoch_time['date'].apply(
                        lambda row: datetime.fromtimestamp(int(row) / 1000).strftime('%Y-%m-%d %H:%M:%S'))
                    df.drop(df.loc[df['frequency'].isna()].index, inplace=True)
                    df = pd.concat([df, epoch_time])

            df.drop(axis=1, inplace=True, labels="accountsPayable") \
                if "accountsPayable" in df.columns.to_list() else None

            df.to_sql(endpoint_name.lower(), con=POSTGRES_URL, index=False, if_exists='append', schema='market')

            # update the db, so we know what historical data we've collected.
            pipeline_metadata = pd.read_sql('SELECT * FROM market.pipeline_metadata', con=POSTGRES_URL)
            pipeline_metadata['last_updated'].loc[pipeline_metadata['endpoint_name'] == endpoint_name] \
                = to_date.strftime('%Y-%m-%d')

            pipeline_metadata.to_sql('pipeline_metadata', con=POSTGRES_URL, index=False, if_exists='replace',
                                     schema='market')

        except Exception:
            print_exc()
            logger.warning(f'Could not insert data for {endpoint_name}.')

        return logger.info(f'{endpoint_name} updated with records from {pull_from}')

    def update_data(self):
        """Pull the latest data for a stock.
        Arguments: stock {str} stock to find number of available records for.
        """
        endpoint_metadata = pd.read_sql('SELECT * FROM market.pipeline_metadata', con=POSTGRES_URL)
        for from_date, endpoint in endpoint_metadata.itertuples(index=False, name=None):
            # there is a huge flaw in that they can only return 5000 data points at a time. We
            # will attempt to iterate by month to back fill data.
            logger.info(f'Updating {endpoint}, pulling all data from {from_date} to present.')
            try:
                while pd.to_datetime(from_date) < datetime.today():
                    to_date = pd.to_datetime(from_date) + pd.DateOffset(months=1)
                    self.ping_endpoint(endpoint, from_date, to_date)
                    from_date = pd.to_datetime(from_date) + pd.DateOffset(months=1)
            except:
                print_exc()

        logger.info('All data has been updated')


self = Pipeline()
