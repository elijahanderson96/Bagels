import os
import requests
from dotenv import load_dotenv
import pandas as pd
from decorators import cast_as_dataframe, write_to_db
import logging
from datetime import date
from dateutil.relativedelta import relativedelta
from config import treasury_endpoints
from prices import UpdatePrices


logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)
load_dotenv()  # take environment variables from .env.


class Iex_Base:
    def __init__(self, production=False):
        if production:
            self.token = os.getenv('PRODUCTION_TOKEN')
            self.base_url = 'https://cloud.iexapis.com/'
        else:
            self.token = os.getenv('SANDBOX_TOKEN')
            self.base_url = 'https://sandbox.iexapis.com/'

        self.version = 'stable'
        self.engine = os.getenv('POSTGRES_CONNECTION')


class Pipeline(Iex_Base):
    def __init__(self, production=False):
        super().__init__(production)
        self.url = self.base_url + self.version + '/time-series/'

    @staticmethod
    @write_to_db
    @cast_as_dataframe
    def timeseries_inventory():
        url = 'https://cloud.iexapis.com/' + 'stable/' + 'time-series'
        r = requests.get(url=url, params={'token': os.getenv("PRODUCTION_TOKEN")})
        return r

    @staticmethod
    @cast_as_dataframe
    def timeseries_metadata(time_series_id=None, stock=None,
                            subkey='ttm'):  # TODO exception handling for incorrect time-series ids and stocks
        """See how many records IEX has to offer for a given time series for a given stock. Passing None
        for both time_series_id and stock gives you all the metadata (all time series for all stocks).
            Parameters:
                time_series_id: The time series endpoint. See get_time_series_inventory for available endpoints
                stock: the symbol in question.
                subkey: a filter parameter for the type of content returned. We'll stick with TTM as default and not annual
                since usually its redundant information.

            Returns:
                r: JSON response as a dataframe (thanks to the decorator)."""
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
        return r

    # Fix the index of this dataframe
    @write_to_db
    @cast_as_dataframe
    def fundamentals(self, stock, subkey='ttm', last=5, interpolate=False):
        assert subkey in ['ttm', 'quarterly'], 'Subkey must be ttm or quarterly'
        url = self.url + f'/FUNDAMENTALS/{stock}/{subkey}'
        logger.info(f'Pinging {url} for fundamentals data')
        r = requests.get(url, params={'last': last, 'token': self.token})
        return r

    @write_to_db
    @cast_as_dataframe
    def treasury_rates(self, symbol, last=5):
        """Only doing 30 year for now, id is dgs30"""
        assert symbol in ['dgs30'], 'Only considering 30 year treasure rates at this time'
        last_n_years = date.today() - relativedelta(years=last)
        url = self.url + f'TREASURY/{symbol}'
        r = requests.get(url, params={'token': self.token, 'from': last_n_years})
        return r

    @staticmethod
    def shares_outstanding(symbol):
        """        GET / stock / {symbol} / stats / {stat?}"""
        url = f'https://cloud.iexapis.com/stable/stock/{symbol}/stats/sharesOutstanding'
        r = requests.get(url, params={'token': os.getenv('PRODUCTION_TOKEN')})
        return r

    def pull_latest(self, stock):
        """This function shows how many records exist for a given symbol 
        and compares how many already exist in our database, such that we can 
        take the difference and only add the newest records that don't exist already. 
        This is desirable as we do not want to pull more than we need, otherwise
        we incur unnecessary cost in our API call.

        Arguments: stock {str} -- Required. This is the stock you're querying to determine
        number of available records for.
        """
        # Probe IEX Cloud to see how many records they have
        metadata = self.timeseries_metadata(time_series_id='FUNDAMENTALS',
                                            stock=stock)
        # Retrieve the "count" entry in the metadata
        n_records = metadata.loc[(metadata['subkey'] == 'TTM')]['count']
        logger.info(f'There were {int(n_records)} TTM records found on IEX Cloud.')
        current_records = pd.read_sql(f"SELECT * FROM Fundamentals WHERE symbol='{stock}';",
                                      self.engine)
        current_records = len(current_records.loc[(current_records['symbol'] == stock)][
                                  'symbol'])
        logger.info(f'We currently have {current_records} records in our database')
        n_records = int(n_records) - int(current_records)  
        logger.info(f'Number of records being fetched for {stock}: {n_records}')

        if n_records != 0:
            logger.info(f'Grabbing latest fundamentals data for {stock} and interpolating')
            self.fundamentals(stock, last=n_records, interpolate=True)
        else:
            logger.info(f'Records are up to date for {stock}')

        # Return the number of records we need to fetch, so that the int is accessible via XCOM (Cross communication,
        # a way to pass parameters between tasks in a DAG)
        return n_records
    
    def run(self, stocks: list):
        for stock in stocks:
            self.pull_latest(stock)
            shares_outstanding = int(self.shares_outstanding(stock).text)
            prices = UpdatePrices(stock, shares_outstanding)
            

if __name__ == '__main__':
    DataGetter = Pipeline()
    DataGetter.run(['C', 'BAC', 'JPM','MS','GS','TSLA','GOOGL','AAPL','KO','MSFT'])

