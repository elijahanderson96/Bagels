import datetime
from traceback import print_exc

from sqlalchemy import create_engine
from yfinance import download

from config.common import *
from data.iex import Iex

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger('urllib3').setLevel(logging.WARNING)


class Prices(Iex):
    """This class is responsible for constructing the stock_prices table."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def resolve_min_date(symbol):
        engine = create_engine(POSTGRES_URL)
        min_date_fundamental_valuations = pd.read_sql(f'SELECT MIN("asOfDate") '
                                                      f'FROM market.fundamental_valuations '
                                                      f"WHERE symbol='{symbol}'",
                                                      engine).squeeze()
        max_date_fundamental_valuations = pd.read_sql(f'SELECT MAX("asOfDate") '
                                                      f'FROM market.fundamental_valuations '
                                                      f"WHERE symbol='{symbol}'",
                                                      engine).squeeze()

        max_date_stock_prices = pd.read_sql(f'SELECT MAX(date) '
                                            f'FROM market.stock_prices '
                                            f"WHERE symbol='{symbol}'",
                                            engine).squeeze()
        engine.dispose()
        return min_date_fundamental_valuations, max_date_fundamental_valuations, max_date_stock_prices

    def fetch_stock_price(self, symbol):
        min_date, max_date, current_date = self.resolve_min_date(symbol)
        logger.info(f'Currently fetching stock prices for {symbol}')
        logger.info(f'Max Date: {max_date}, Min Date: {min_date}, Currently in DB: {current_date}')

        if current_date:
            df = download(symbol, group_by='ticker', auto_adjust=True, start=current_date + datetime.timedelta(days=1),
                          progress=False)
        else:
            df = download(symbol, group_by='ticker', auto_adjust=True, start='2012-01-01', progress=False)

        df['symbol'] = symbol
        shares_outstanding = super().shares_outstanding(symbol)
        if shares_outstanding == 0: return
        df['shares_outstanding'] = shares_outstanding
        df['marketCap'] = df['Close'] * df['shares_outstanding']
        df.reset_index(inplace=True)
        df.rename(columns={column: column.lower() for column in df.columns}, inplace=True)
        df.drop(labels=['open', 'high', 'low', 'volume'], axis=1, inplace=True)
        return df

    def update_db(self):
        for symbol in reversed(SYMBOLS):
            prices_matrix = self.fetch_stock_price(symbol)
            if isinstance(prices_matrix, pd.DataFrame):
                try:
                    prices_matrix.to_sql('stock_prices', con=POSTGRES_URL, schema='market', if_exists='append',
                                         index=False)
                    logger.info(f'Inserted {prices_matrix.shape[0]} rows for {symbol} in stock prices')
                except:
                    print_exc()
                    logger.warning('Could not insert stock prices into database. Likely a key error')


if __name__ == '__main__':
    obj = Prices()
    obj.update_db()
