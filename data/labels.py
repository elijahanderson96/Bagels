import logging
from yfinance import download
from config.common import *
from data.iex import Iex
from time import sleep
from config.mappings import *

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger('urllib3').setLevel(logging.WARNING)


class Prices(Iex):
    """This class is responsible for constructing the stocks and prices tables.
    The prices table is updated daily. THe stocks table contains metadata regarding
    what sector a stock is in."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def resolve_min_date(symbol):
        min_date_fundamentals = pd.read_sql(f'SELECT MIN(reportdate) '
                                            f'FROM market.fundamentals '
                                            f"WHERE symbol='{symbol}'",
                                            POSTGRES_URL).squeeze()
        max_date_fundamentals = pd.read_sql(f'SELECT MAX(reportdate) '
                                            f'FROM market.fundamentals '
                                            f"WHERE symbol='{symbol}'",
                                            POSTGRES_URL).squeeze()
        max_date_stock_prices = pd.read_sql(f'SELECT MAX(date) '
                                            f'FROM market.stock_prices '
                                            f"WHERE symbol='{symbol}'",
                                            POSTGRES_URL).squeeze()

        return min_date_fundamentals, max_date_fundamentals, max_date_stock_prices

    def fetch_stock_price(self, symbol):
        sleep(1)
        min_date, max_date, current_date = self.resolve_min_date(symbol)
        logger.info(f'Currently fetching stock prices for {symbol}')
        logger.info(f'Max Date: {max_date}, Min Date: {min_date}, Currently in DB: {current_date}')

        if current_date:
            df = download(symbol, group_by='ticker', auto_adjust=True, threads=True,
                          start=current_date, progress=False)
        else:
            df = download(symbol, group_by='ticker', auto_adjust=True, threads=True, start='2000-01-01', progress=False)

        df['symbol'] = symbol
        df['shares_outstanding'] = super()._shares_outstanding(symbol)
        df['marketCap'] = df['Close'] * df['shares_outstanding']
        df.reset_index(inplace=True)
        df.rename(columns={column: column.lower() for column in df.columns}, inplace=True)
        df.drop(labels=['open', 'high', 'low', 'volume'], axis=1, inplace=True)
        return df

    def update_db(self):
        for symbol in SYMBOLS:
            prices_matrix = self.fetch_stock_price(symbol)
            try:
                prices_matrix.to_sql('stock_prices', con=POSTGRES_URL, schema='market', if_exists='append', index=False)
                logger.info(f'Inserted {prices_matrix.shape[0]} rows for {symbol} in stock prices')
            except:
                logger.warning('Could not insert stock prices into database. Likely a key error')


if __name__ == '__main__':
    obj = Prices()
    obj.update_db()
