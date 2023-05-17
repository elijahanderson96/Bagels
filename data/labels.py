import datetime
import logging
from traceback import print_exc

import pandas as pd
import requests
from sqlalchemy import create_engine

from config.configs import POSTGRES_URL
from data.iex import Iex

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger('urllib3').setLevel(logging.WARNING)


class Prices(Iex):
    """This class is responsible for constructing the stock_prices table."""

    def __init__(self):
        super().__init__()

    def historical_stock_prices(self, stock, from_date, to_date=datetime.datetime.today().strftime('%Y-%m-%d')):
        logger.info(f'Grabbing stock prices for {stock} from {from_date} to {to_date}')
        url = self.url + f'HISTORICAL_PRICES/{stock}'
        logger.info(f'Pinging {url} for energy data')
        r = requests.get(url, params={'from': from_date, 'to': to_date, 'token': self.token})
        df = self.json_to_dataframe(r)
        return df

    @staticmethod
    def resolve_min_date(symbol):
        engine = create_engine(POSTGRES_URL)
        max_date_stock_prices = pd.read_sql(f'''SELECT MAX("priceDate") '''
                                            f'FROM bagels.labels '
                                            f"WHERE symbol='{symbol}'",
                                            engine).squeeze()
        engine.dispose()
        return max_date_stock_prices

    def fetch_stock_price(self, symbol):
        from_date = self.resolve_min_date(symbol)
        logger.info(f'Currently fetching stock prices for {symbol}')
        logger.info(f'Min Date: {from_date}')

        df = self.historical_stock_prices(symbol, from_date=from_date + datetime.timedelta(days=1)
        if from_date else '01-01-2000')

        return df

    def update_db(self):
        fundamental_valuations_symbols = pd.read_sql("SELECT DISTINCT(symbol) "
                                                     "FROM bagels.fundamental_valuations;",
                                                     con=POSTGRES_URL)['symbol']
        for symbol in fundamental_valuations_symbols:
            prices_matrix = self.fetch_stock_price(symbol)
            if isinstance(prices_matrix, pd.DataFrame):
                try:
                    prices_matrix.to_sql('labels', con=POSTGRES_URL, schema='bagels', if_exists='append',
                                         index=False)
                    logger.info(f'Inserted {prices_matrix.shape[0]} rows for {symbol} in stock prices')
                except:
                    print_exc()
                    logger.warning('Could not insert stock prices into database. Likely a key error')


if __name__ == '__main__':
    obj = Prices()
    obj.update_db()
